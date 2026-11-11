#!/usr/bin/env python3
"""Create or apply an exact sparse patch for a numeric NPZ or flat PyTorch state dict."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import tempfile
import zipfile

import numpy as np


FORMAT = "wsp.checkpoint_patch.v1"
TORCH_SUFFIXES = {".pt", ".pth"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


class Checkpoint:
    def __init__(self, path, expected_hash=None):
        self.format = "npz" if path.suffix == ".npz" else "torch"
        require(path.suffix == ".npz" or path.suffix in TORCH_SUFFIXES, "unsupported checkpoint suffix")
        self.torch = None
        with path.open("rb") as stream:
            self.file_hash = sha256(stream)
            require(expected_hash is None or self.file_hash == expected_hash, "base checkpoint SHA-256 mismatch")
            stream.seek(0)
            if self.format == "npz":
                with np.load(stream, allow_pickle=False) as archive:
                    require(len(archive.files) == len(set(archive.files)), "duplicate NPZ keys")
                    require("file" not in archive.files, "NPZ key 'file' is reserved")
                    self.values = {key: archive[key].copy(order="K") for key in archive.files}
            else:
                import torch
                self.torch = torch
                self.values = torch.load(stream, map_location="cpu", weights_only=True)
                require(isinstance(self.values, dict), "expected a flat PyTorch state dict")
                self.check_torch_layout()
        require(self.values and all(isinstance(k, str) for k in self.values), "expected named checkpoint entries")
        self.metadata = encode(getattr(self.values, "_metadata", None))

    def check_torch_layout(self):
        storage = set()
        for key, value in self.values.items():
            require(type(value) is self.torch.Tensor and value.layout == self.torch.strided
                    and not value.is_quantized and not value.requires_grad and value.is_contiguous()
                    and not value.storage_offset() and not value.is_conj() and not value.is_neg(),
                    f"{key}: require a plain contiguous tensor")
            if value.numel():
                address = value.untyped_storage().data_ptr()
                require(address not in storage, f"{key}: shared tensor storage needs a native adapter")
                storage.add(address)

    def array(self, key):
        value = self.values[key]
        if self.torch is not None:
            if value.dtype == self.torch.bfloat16:
                return value.view(self.torch.int16).numpy()
            return value.numpy()
        require(not value.dtype.hasobject and value.dtype.fields is None, f"{key}: unsupported NPZ dtype")
        return value

    def spec(self, key):
        value = self.values[key]
        dtype = str(value.dtype) if self.torch is not None else value.dtype.str
        return {"name": key, "dtype": dtype, "shape": list(value.shape)}

    def elements(self, key):
        value = self.array(key)
        return value.reshape(-1).view(np.uint8).reshape(value.size, value.itemsize)

    def state_hash(self):
        entries = [dict(**self.spec(key), bytes_sha256=hashlib.sha256(self.elements(key)).hexdigest())
                   for key in sorted(self.values)]
        return hashlib.sha256(encode({"entries": entries, "metadata": json.loads(self.metadata)})).hexdigest()

    def check_editable(self, key):
        value = self.values[key]
        floating = value.is_floating_point() if self.torch is not None else value.dtype.kind == "f"
        require(floating, f"{key}: only floating-point entries may change")

    def check_finite(self, key):
        value = self.values[key]
        finite = self.torch.isfinite(value) if self.torch is not None else np.isfinite(value)
        require(bool(finite.all()), f"{key}: patched tensor contains nonfinite values")

    def save(self, stream):
        if self.torch is None:
            np.savez_compressed(stream, **self.values)
        else:
            self.torch.save(self.values, stream)


def write_new(path, write):
    # Publish the completed file atomically, with no replacement of existing paths.
    fd, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            write(stream)
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def create(base_path, candidate_path, patch_path):
    base, candidate = Checkpoint(base_path), Checkpoint(candidate_path)
    require(base.format == candidate.format, "checkpoint formats differ")
    require(base.values.keys() == candidate.values.keys() and base.metadata == candidate.metadata,
            "checkpoint keys or metadata differ")
    edits, payload = [], {}
    for key in sorted(base.values):
        spec = base.spec(key)
        require(spec == candidate.spec(key), f"{key}: dtype or shape differs")
        a, b = base.elements(key), candidate.elements(key)
        index = np.flatnonzero(np.any(a != b, axis=1))
        if not index.size:
            continue
        candidate.check_editable(key)
        candidate.check_finite(key)
        i = len(edits)
        edits.append(dict(spec, changed=int(index.size)))
        payload[f"index_{i}"] = index.astype("<u4" if a.shape[0] <= 2**32 else "<u8")
        payload[f"value_{i}"] = b[index]
    require(edits, "checkpoints have no changed entries")
    manifest = dict(format=FORMAT, checkpoint_format=base.format, base_sha256=base.file_hash,
                    candidate_sha256=candidate.file_hash, candidate_state_sha256=candidate.state_hash(),
                    edits=edits)
    payload["manifest"] = np.frombuffer(encode(manifest), dtype=np.uint8)
    write_new(patch_path, lambda stream: np.savez_compressed(stream, **payload))
    return manifest


def apply(base_path, patch_path, output_path):
    with np.load(patch_path, allow_pickle=False) as patch:
        header = patch["manifest"]
        require(header.dtype == np.uint8 and header.ndim == 1, "invalid patch manifest array")
        manifest = json.loads(header.tobytes())
        require(manifest["format"] == FORMAT, "unsupported patch format")
        base_hash = manifest["base_sha256"]
        require(isinstance(base_hash, str) and re.fullmatch("[0-9a-f]{64}", base_hash),
                "invalid base checkpoint SHA-256")
        base = Checkpoint(base_path, base_hash)
        suffixes = {".npz"} if base.format == "npz" else TORCH_SUFFIXES
        require(base.format == manifest["checkpoint_format"] and output_path.suffix in suffixes,
                "checkpoint/output format mismatch")
        edits = manifest["edits"]
        names = [row["name"] for row in edits]
        require(names and len(names) == len(set(names)), "empty or duplicate edit keys")
        expected_arrays = {"manifest"} | {f"{prefix}_{i}" for i in range(len(edits)) for prefix in ("index", "value")}
        require(len(patch.files) == len(expected_arrays) and set(patch.files) == expected_arrays,
                "unexpected or missing patch arrays")
        for i, row in enumerate(edits):
            name = row["name"]
            spec = base.spec(name)
            require(all(row[k] == v for k, v in spec.items()), f"{name}: patch dtype or shape mismatch")
            base.check_editable(name)
            if base.torch is None:
                base.values[name] = np.ascontiguousarray(base.values[name]).reshape(spec["shape"])
            data = base.elements(name)
            index, values = patch[f"index_{i}"], patch[f"value_{i}"]
            require(index.dtype in (np.dtype("<u4"), np.dtype("<u8")) and index.ndim == 1
                    and 0 < index.size == row["changed"] and int(index[-1]) < data.shape[0]
                    and bool((index[1:] > index[:-1]).all()), f"{name}: invalid patch indices")
            require(values.dtype == np.uint8 and values.shape == (index.size, data.shape[1]),
                    f"{name}: invalid replacement values")
            require(bool(np.any(data[index] != values, axis=1).all()), f"{name}: no-op replacement")
            data[index] = values
            base.check_finite(name)
    require(base.state_hash() == manifest["candidate_state_sha256"], "reconstructed state SHA-256 mismatch")
    write_new(output_path, base.save)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command, second, third in (("create", "candidate", "patch"), ("apply", "patch", "output")):
        entry = sub.add_parser(command)
        for name in ("base", second, third):
            entry.add_argument(name, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "create":
            manifest = create(args.base, args.candidate, args.patch)
        else:
            manifest = apply(args.base, args.patch, args.output)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, ImportError,
            pickle.UnpicklingError, zipfile.BadZipFile, EOFError) as error:
        parser.exit(1, f"checkpoint patch: {error}\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
