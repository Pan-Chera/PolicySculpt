"""Offline probe for the existing Cube2 visual ACT native inference protocol."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from chunks import audit_chunks
from pairs import reference
from traces import read


def predict_packets(job, packets):
    import numpy as np
    import torch

    job = Path(job).resolve()
    spec = read(job / "plan.json")
    source = job / "source/learn.py"
    model_path = job / "model.pt"
    train = read(job / "training.json")
    model_ref = reference(model_path)
    if model_ref["sha256"] != train["checkpoint_sha256"]:
        raise ValueError("visual checkpoint differs from training receipt")
    sys.path.insert(0, str(source.parent))
    module_spec = importlib.util.spec_from_file_location("native_visual_act", source)
    native = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(native)
    torch.set_num_threads(2)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = native.CubeACT().eval()
    model.load_state_dict(checkpoint["model"])
    sm, ss, am, ast = checkpoint["normalization"]
    rgb_shape = (1, 2, spec["camera_height"], spec["camera_width"], 3)
    length = 200 + int(np.prod(rgb_shape))
    if not packets or len(packets) % length:
        raise ValueError("invalid visual inference packet length")
    predictions = []
    with torch.inference_mode():
        for index in range(0, len(packets), length):
            raw = memoryview(packets)[index:index+length]
            context = np.frombuffer(raw, dtype="<f4", count=2)
            state = torch.from_numpy(np.frombuffer(raw, dtype="<f4", count=48, offset=8).copy()).reshape(1, 3, 16)
            rgb = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8, offset=200).copy()).reshape(rgb_shape)
            output, _ = model(rgb, (state-sm)/ss, torch.tensor([int(context[0])]), torch.tensor([context[1]]))
            predictions.append((output[0]*ast+am).numpy())
    return predictions, model_ref, train["parameters"]


def probe(job, sample, runtime):
    import numpy as np

    job = Path(job).resolve()
    spec = read(job / "plan.json")
    source = job / "source/learn.py"
    sample = Path(sample).resolve()
    with np.load(sample, allow_pickle=False) as data:
        images, states = data["rgb"], data["state"]
        instruction = int(data["instruction"])
    stride, prefix = spec["stride"], spec["replan"]
    ticks = range(0, min(len(images)-prefix+1, prefix*3), prefix)
    if len(ticks) < 2:
        raise ValueError("need at least two full returned prefixes in the trace")
    requests = []
    for tick in ticks:
        rgb = np.ascontiguousarray(images[tick], dtype=np.uint8)
        history = np.ascontiguousarray(states[np.maximum(tick-np.arange(2, -1, -1), 0)], dtype="<f4")
        context = np.asarray([instruction, tick*stride/2400], dtype="<f4")
        requests.append(context.tobytes() + history.tobytes() + rgb.tobytes())
    packets = b"".join(requests)
    predictions, model_ref, parameters = predict_packets(job, packets)
    raw = subprocess.run([runtime, "-B", str(source), "infer"], input=packets,
                         check=True, capture_output=True, timeout=60).stdout
    returned = np.frombuffer(raw, dtype="<f4").reshape(len(ticks), prefix, 16)
    rows = []
    for i, tick in enumerate(ticks):
        full = predictions[i]
        if full.shape != (spec["chunk"], 16):
            raise ValueError("unexpected action chunk shape")
        for index, action in enumerate(returned[i]):
            row = {"tick": tick+index, "sent": {"id": str(i), "index": index, "action": action.tolist()}}
            if index == 0:
                row["generated"] = {"id": str(i), "actions": full.tolist()}
            rows.append(row)
    return {"mode": "fixed_recorded_observations", "checkpoint": model_ref,
            "native_source": reference(source), "input": reference(sample),
            "rgb_shape": list(images.shape[1:]), "parameters": parameters,
            "configured_replan": prefix, "physics_steps_per_action": stride,
            "max_prefix_error": float(max(np.max(np.abs(full[:prefix]-actual))
                                         for full, actual in zip(predictions, returned))),
            "audit": audit_chunks(rows, tolerance=1e-5)}


if __name__ == "__main__":
    predictions, model, count = predict_packets(sys.argv[1], sys.stdin.buffer.read())
    print(json.dumps({"predictions": [value.tolist() for value in predictions],
                      "checkpoint": model, "parameters": count}, allow_nan=False))
