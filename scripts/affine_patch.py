#!/usr/bin/env python3
"""Fold a fixed affine intervention into one linear layer, using NumPy only.

Column convention: y = W h + b. This never loads .pt/pickle, fits data,
executes a policy, or contacts hardware. Numeric checks use controlled float64
arithmetic with casts at layer/intervention boundaries, not a hardware emulator.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


class Invalid(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise Invalid(message)


def floating(value, name, ndim):
    array = np.asarray(value)
    require(array.ndim == ndim and array.size > 0, f"{name}: invalid rank/empty")
    require(array.dtype.kind == "f" and array.dtype.itemsize in (2, 4, 8),
            f"{name}: require real float16/32/64, no objects/integers/complex")
    require(bool(np.isfinite(array).all()), f"{name}: nonfinite value")
    return array


def contract(weight, bias, projection, offset, side):
    require(side in ("input", "output"), "side must be input or output")
    weight = floating(weight, "weight", 2)
    if bias is not None:
        bias = floating(bias, "bias", 1)
        require(bias.shape == (weight.shape[0],), "bias shape mismatch")
        require(bias.dtype == weight.dtype, "mixed layer dtypes require a runtime-specific adapter")
    dimension = weight.shape[1 if side == "input" else 0]
    projection = floating(projection, "P", 2)
    offset = floating(offset, "offset", 1)
    require(projection.shape == (dimension, dimension), "P shape/axis mismatch")
    require(offset.shape == (dimension,), "offset shape mismatch")
    return weight, bias, projection, offset


def fold_affine(weight, bias, projection, offset, side):
    weight, bias, projection, offset = contract(weight, bias, projection, offset, side)
    w = weight.astype(np.float64)
    b = np.zeros(w.shape[0]) if bias is None else bias.astype(np.float64)
    p, c = projection.astype(np.float64), offset.astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        if side == "input":
            new_w, new_b = w @ p, b + w @ c
        else:
            new_w, new_b = p @ w, p @ b + c
        if bias is None:
            require(bool(np.all(new_b == 0)), "would require adding a new bias parameter")
        new_w = new_w.astype(weight.dtype)
        new_b = None if bias is None else new_b.astype(bias.dtype)
    require(bool(np.isfinite(new_w).all()), "weight overflow after folding/cast")
    require(new_b is None or bool(np.isfinite(new_b).all()), "bias overflow after folding/cast")
    return new_w, new_b


def edit_output_writers(weight, direction, columns, alpha):
    """Subtract a unit output direction only from selected input columns.

    This is not full concept erasure: bias and other writers stay unchanged.
    """
    weight = floating(weight, "weight", 2)
    u = floating(direction, "direction", 1).astype(np.float64)
    require(u.shape == (weight.shape[0],), "output direction axis mismatch")
    require(abs(float(u @ u) - 1) <= 1e-6, "direction must be explicitly unit-normalized")
    require(isinstance(columns, (list, tuple)) and bool(columns), "empty columns")
    require(all(type(i) is int and 0 <= i < weight.shape[1] for i in columns), "invalid column index")
    require(len(set(columns)) == len(columns), "duplicate columns")
    require(isinstance(alpha, (int, float)) and not isinstance(alpha, bool) and np.isfinite(alpha),
            "invalid alpha")
    result = weight.copy()
    selected = weight[:, columns].astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        result[:, columns] = selected - float(alpha) * np.outer(u, u @ selected)
    require(bool(np.isfinite(result).all()), "writer edit overflow")
    return result


def linear(features, weight, bias, dtype):
    result = features.astype(np.float64) @ weight.astype(np.float64).T
    if bias is not None:
        result += bias.astype(np.float64)
    return result.astype(dtype)


def compare_intervention(weight, bias, projection, offset, side, features, new_w, new_b, atol, rtol):
    weight, bias, projection, offset = contract(weight, bias, projection, offset, side)
    for name, value in (("atol", atol), ("rtol", rtol)):
        require(isinstance(value, (int, float)) and not isinstance(value, bool)
                and np.isfinite(value) and value >= 0, f"invalid {name}")
    features = floating(features, "features", 2)
    require(features.shape[1] == weight.shape[1], "features must be original layer input rows")
    with np.errstate(over="ignore", invalid="ignore"):
        x = features.astype(weight.dtype)
        if side == "input":
            transformed = (x.astype(np.float64) @ projection.astype(np.float64).T + offset).astype(weight.dtype)
            expected = linear(transformed, weight, bias, weight.dtype)
        else:
            original = linear(x, weight, bias, weight.dtype)
            expected = (original.astype(np.float64) @ projection.astype(np.float64).T + offset).astype(weight.dtype)
        actual = linear(x, new_w, new_b, weight.dtype)
    require(bool(np.isfinite(expected).all() and np.isfinite(actual).all()),
            "nonfinite reference/candidate output")
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    no_op = bool(np.array_equal(weight, new_w)
                 and ((bias is None and new_b is None) or np.array_equal(bias, new_b)))
    return dict(equivalent_on_given_features=bool(np.allclose(actual, expected, atol=atol, rtol=rtol)),
                max_abs_error=float(error.max()), mean_abs_error=float(error.mean()),
                feature_rows=int(features.shape[0]), no_op_after_cast=no_op,
                linear_layer_frame_equivalents=int(2 * features.shape[0]),
                changed_weight_entries=int(np.count_nonzero(weight != new_w)),
                changed_bias_entries=0 if bias is None else int(np.count_nonzero(bias != new_b)),
                reference_arithmetic="float64-with-boundary-casts-not-hardware-emulation")


def load_npz(path, allowed, required):
    require(Path(path).suffix == ".npz", "require .npz files, no pickle/checkpoint loading")
    with np.load(path, allow_pickle=False) as data:
        require(set(data.files) <= set(allowed) and set(required) <= set(data.files),
                f"unexpected/missing arrays in {Path(path).name}")
        return {key: data[key].copy() for key in data.files}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", required=True, help="npz: weight [out,in], optional bias [out]")
    parser.add_argument("--transform", required=True, help="npz: P, offset (column convention)")
    parser.add_argument("--features", required=True, help="npz: features [N,in], development data")
    parser.add_argument("--side", choices=["input", "output"], required=True)
    parser.add_argument("--atol", type=float, required=True)
    parser.add_argument("--rtol", type=float, required=True)
    parser.add_argument("--out", required=True, help="new layer npz, never overwrites")
    parser.add_argument("--report", required=True, help="new JSON report, never overwrites")
    args = parser.parse_args()
    report_path, out = Path(args.report), Path(args.out)
    report = dict(status="rejected", model_executed=False, hardware_executed=False,
                  full_policy_executed=False,
                  physical_effectiveness_verified=False)
    may_write_report = False
    try:
        paths = [Path(p).resolve() for p in [args.layer, args.transform, args.features, args.out, args.report]]
        require(len(paths) == len(set(paths)), "input/output paths must be distinct")
        require(not report_path.exists(), "report exists; refusing overwrite")
        may_write_report = True
        require(not out.exists(), "output exists; refusing overwrite")
        require(out.suffix == ".npz", "output must end in .npz")
        layer = load_npz(args.layer, {"weight", "bias"}, {"weight"})
        transform = load_npz(args.transform, {"P", "offset"}, {"P", "offset"})
        features = load_npz(args.features, {"features"}, {"features"})["features"]
        weight, bias = layer["weight"], layer.get("bias")
        p, c = transform["P"], transform["offset"]
        new_w, new_b = fold_affine(weight, bias, p, c, args.side)
        if isinstance(features, np.ndarray) and features.ndim == 2:
            report["reference_layer_frame_equivalents_reserved"] = int(2 * features.shape[0])
        report.update(compare_intervention(weight, bias, p, c, args.side, features, new_w, new_b,
                                           args.atol, args.rtol))
        report.update(side=args.side, dtype=str(weight.dtype), atol=args.atol, rtol=args.rtol,
                      input_sha256={"layer": sha(args.layer), "transform": sha(args.transform),
                                    "features": sha(args.features)})
        require(report["equivalent_on_given_features"], "equivalence tolerance failed; no output layer written")
        arrays = {"weight": new_w}
        if new_b is not None:
            arrays["bias"] = new_b
        with out.open("xb") as stream:
            np.savez(stream, **arrays)
        report.update(status="local_affine_check_only", output_sha256=sha(out))
    except (ValueError, TypeError, OSError, KeyError, OverflowError) as exc:
        report["error"] = str(exc)
    if may_write_report:
        try:
            with report_path.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
                stream.write("\n")
        except OSError as exc:
            report.update(status="rejected", report_write_error=str(exc))
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0 if report["status"] == "local_affine_check_only" else 1


if __name__ == "__main__":
    sys.exit(main())
