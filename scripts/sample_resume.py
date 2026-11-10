"""Store completed evaluation pairs and resume missing samples from a frozen plan."""

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


FORMAT = "wsp.sample_resume.v1"
ROLES = {"baseline", "candidate", "evaluator"}


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def fingerprint(value):
    """Hash a JSON-compatible value. Include exact native state bytes by their file or byte hash."""
    return hashlib.sha256(json_bytes(value)).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def publish(path, value):
    """Publish a complete file without replacing a committed result."""
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(json_bytes(value) + b"\n")
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def read(path):
    return json.loads(path.read_bytes())


class SampleJournal:
    """A single coordinator owns execution and commits one complete pair per sample."""

    def __init__(self, directory, identities, samples):
        if set(identities) != ROLES or not all(is_digest(h) for h in identities.values()):
            raise ValueError("require baseline, candidate, and evaluator SHA-256 identities")
        if not samples or not all(isinstance(k, str) and k and is_digest(v) for k, v in samples.items()):
            raise ValueError("require stable nonempty sample IDs with SHA-256 input identities")
        self.samples = dict(samples)
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=True)
        self.plan = {"format": FORMAT, "identities": dict(identities), "samples": self.samples}
        self.plan_hash = fingerprint(self.plan)
        plan_path = self.directory / "plan.json"
        try:
            publish(plan_path, self.plan)
        except FileExistsError:
            previous = read(plan_path)
            if previous != self.plan:
                raise ValueError("frozen evaluation identity mismatch; use a separate run directory") from None

    def pair_path(self, sample_id):
        sample_key = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
        return self.directory / f"pair-{sample_key}.json"

    def binding(self, sample_id):
        return {"plan_sha256": self.plan_hash, "sample_id": sample_id,
                "input_sha256": self.samples[sample_id]}

    def completed(self):
        """Return only complete, verified pairs. Refuse inconsistent committed receipts."""
        results = {}
        for sample_id in self.samples:
            try:
                record = read(self.pair_path(sample_id))
            except FileNotFoundError:
                continue
            body = record["pair"]
            if (body["binding"] != self.binding(sample_id) or
                    record["sha256"] != fingerprint(body) or
                    not isinstance(body["baseline"], dict) or not isinstance(body["candidate"], dict)):
                raise ValueError(f"inconsistent paired receipt for {sample_id}")
            results[sample_id] = {"baseline": body["baseline"], "candidate": body["candidate"]}
        return results

    def commit(self, sample_id, baseline, candidate):
        """Commit two valid results. The caller decides whether an outcome is technically valid."""
        if not isinstance(baseline, dict) or not isinstance(candidate, dict):
            raise ValueError("a completed pair requires both result dictionaries")
        body = {"binding": self.binding(sample_id), "baseline": baseline, "candidate": candidate}
        publish(self.pair_path(sample_id), {"pair": body, "sha256": fingerprint(body)})
