"""Command-line interface for robot traces, paired evaluation and action chunks."""
import argparse
import json
from pathlib import Path

from adapters import cube_manifest, cube_simulation_manifest, goblet_manifest
from chunks import audit_chunks
from pairs import run
from traces import diagnose, episode, jsonl, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    trace = commands.add_parser("diagnose")
    trace.add_argument("base")
    trace.add_argument("candidate")
    trace.add_argument("--format", choices=("goblet", "cube", "canonical"), default="canonical")
    trace.add_argument("--threshold", action="append", default=[], metavar="SIGNAL=VALUE")
    replay = commands.add_parser("evaluate")
    replay.add_argument("manifest")
    replay.add_argument("output")
    replay.add_argument("--limit-arms", type=int)
    archive = commands.add_parser("recorded-manifest")
    archive.add_argument("task", choices=("goblet", "cube"))
    archive.add_argument("repo", type=Path)
    native = commands.add_parser("cube-simulation-manifest")
    native.add_argument("repo", type=Path)
    native.add_argument("case_id")
    native.add_argument("--runtime", type=Path, required=True)
    chunks = commands.add_parser("audit-chunks")
    chunks.add_argument("trace")
    vision = commands.add_parser("visual-act")
    vision.add_argument("job")
    vision.add_argument("sample")
    vision.add_argument("--runtime", required=True)
    visual_sim = commands.add_parser("visual-act-sim")
    visual_sim.add_argument("job")
    visual_sim.add_argument("--actor-runtime", required=True)
    visual_sim.add_argument("--groups", type=int, default=3)
    args = parser.parse_args()

    if args.command == "diagnose":
        thresholds = {}
        for value in args.threshold:
            name, limit = value.split("=", 1)
            thresholds[name] = float(limit)
        result = diagnose(episode(args.base, args.format), episode(args.candidate, args.format), thresholds)
    elif args.command == "recorded-manifest":
        build = goblet_manifest if args.task == "goblet" else cube_manifest
        result = build(args.repo.resolve())
    elif args.command == "cube-simulation-manifest":
        result = cube_simulation_manifest(args.repo.resolve(), args.runtime.absolute(), args.case_id)
    elif args.command == "evaluate":
        result = run(read(args.manifest), args.output, args.limit_arms)
    elif args.command == "audit-chunks":
        result = audit_chunks(list(jsonl(args.trace)))
    elif args.command == "visual-act":
        from visual_act import probe
        result = probe(args.job, args.sample, args.runtime)
    else:
        from visual_rollout import probe
        result = probe(args.job, args.actor_runtime, args.groups)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
