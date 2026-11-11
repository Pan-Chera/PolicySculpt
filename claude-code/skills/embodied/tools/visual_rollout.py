"""Bounded Cube2 visual ACT simulator probe with native controller readback."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from chunks import audit_chunks
from pairs import reference
from traces import difference, read


def probe(job, actor_runtime, groups):
    import numpy as np

    job = Path(job).resolve()
    spec = read(job / "plan.json")
    stride, prefix = spec["stride"], spec["replan"]
    horizon = groups * stride * prefix
    if groups < 1 or horizon > spec["max_episode_ticks"]:
        raise ValueError("probe length must fit inside the native episode horizon")
    source = job / "source"
    sys.path.insert(0, str(source))
    module_spec = importlib.util.spec_from_file_location("native_visual_sim", source / "sim_worker.py")
    native = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(native)
    packets, responses, substeps = [], [], []

    class ObservedSensors(native.Sensors):
        def apply(self, target):
            super().apply(target)
            substeps.append({"interpolated_command": target.tolist(), "applied_command": self.action().tolist()})

    def observe_write(stream, value):
        packets.append(value)
        return stream.write(value)

    def observe_read(stream, count):
        raw = stream.read(count)
        responses.append(raw)
        return raw

    native.Sensors = ObservedSensors
    native.MAX_TICKS = horizon
    model = reference(job / "model.pt")
    if model["sha256"] != read(job / "training.json")["checkpoint_sha256"]:
        raise ValueError("visual checkpoint differs from the original training receipt")
    with subprocess.Popen([actor_runtime, "-B", str(source / "learn.py"), "infer"],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE) as actor:
        observed = SimpleNamespace(
            stdin=SimpleNamespace(write=lambda value: observe_write(actor.stdin, value), flush=actor.stdin.flush),
            stdout=SimpleNamespace(read=lambda count: observe_read(actor.stdout, count)),
        )
        result, _, _ = native.rollout(observed, spec["test_poses"][0], spec["test_scrambles"][0], native.MOVES[0])
        actor.stdin.close()
        if actor.wait(timeout=15):
            raise ValueError("visual inference subprocess exited unsuccessfully")
    if result.get("error") or len(substeps) != horizon or len(packets) != groups:
        raise ValueError(f"incomplete simulator probe: {result}")

    reconstructed = subprocess.run([actor_runtime, "-B", str(Path(__file__).with_name("visual_act.py")), str(job)],
                                   input=b"".join(packets), capture_output=True, check=True, timeout=60)
    full = json.loads(reconstructed.stdout)
    if full["checkpoint"] != model:
        raise ValueError("visual inference and reconstruction checkpoint mismatch")
    rows = []
    prefix_error = 0.0
    for i, raw in enumerate(responses):
        generated = full["predictions"][i]
        returned = np.frombuffer(raw, dtype="<f4").reshape(prefix, 16)
        for index, action in enumerate(returned.tolist()):
            tick = i*prefix+index
            applied = substeps[(tick+1)*stride-1]["applied_command"]
            row = {"tick": tick, "sent": {"id": str(i), "index": index, "action": action},
                   "applied": {"id": str(i), "index": index, "action": applied}}
            if index == 0:
                row["generated"] = {"id": str(i), "actions": generated}
            rows.append(row)
            prefix_error = max(prefix_error, difference(generated[index], action)[0])
    if prefix_error > 1e-5:
        raise ValueError(f"native visual inference differs from full chunk reconstruction: {prefix_error}")
    limit = max(difference(row["interpolated_command"], row["applied_command"])[0] for row in substeps)
    return {"mode": "bounded_native_simulation", "checkpoint": model, "parameters": full["parameters"],
            "native_evaluator": reference(source / "sim_worker.py"), "physics_ticks": result["ticks"],
            "configured_replan": prefix, "physics_steps_per_action": stride, "max_prefix_error": prefix_error,
            "max_controller_projection_difference": limit, "task_outcome_scored": False,
            "audit": audit_chunks(rows, tolerance=1e-5), "controller_substeps": substeps}
