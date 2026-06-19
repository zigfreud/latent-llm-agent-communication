import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import torch


REQUIRED_MANIFEST_FIELDS = {
    "bundle_format",
    "schema_version",
    "input_dim",
    "output_dim",
    "shards",
}
EXPECTED_BUNDLE_FORMAT = "lip_latent_bundle"
EXPECTED_SCHEMA_VERSION = 1


def fail(message):
    raise SystemExit(f"Invalid latent bundle: {message}")


def read_manifest(bundle_dir):
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        fail(f"missing {manifest_path}")

    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except json.JSONDecodeError as exc:
        fail(f"manifest.json is not valid JSON: {exc}")

    if not isinstance(manifest, dict):
        fail("manifest.json must contain a JSON object")

    missing = sorted(REQUIRED_MANIFEST_FIELDS.difference(manifest))
    if missing:
        fail(f"manifest.json missing required field(s): {', '.join(missing)}")

    if manifest["bundle_format"] != EXPECTED_BUNDLE_FORMAT:
        fail(f"bundle_format must be {EXPECTED_BUNDLE_FORMAT}")

    if manifest["schema_version"] != EXPECTED_SCHEMA_VERSION:
        fail(f"schema_version must be {EXPECTED_SCHEMA_VERSION}")

    for field in ("input_dim", "output_dim"):
        if not isinstance(manifest[field], int) or manifest[field] <= 0:
            fail(f"{field} must be a positive integer")

    if not isinstance(manifest["shards"], list) or not manifest["shards"]:
        fail("shards must be a non-empty list")

    if not any(
        isinstance(shard, dict) and shard.get("path") == "shards/shard_0.pt"
        for shard in manifest["shards"]
    ):
        fail("shards must list required shard path: shards/shard_0.pt")

    return manifest


def normalize_shard_path(path_value):
    if not isinstance(path_value, str) or not path_value:
        fail("each shard entry must include a non-empty string path")

    shard_path = PurePosixPath(path_value)
    if shard_path.is_absolute() or ".." in shard_path.parts:
        fail(f"shard path must be relative and stay inside the bundle: {path_value}")

    if len(shard_path.parts) < 2 or shard_path.parts[0] != "shards":
        fail(f"shard path must be under shards/: {path_value}")

    return shard_path


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_shard(path):
    try:
        return torch.load(path, map_location="cpu")
    except Exception as exc:
        fail(f"failed to load shard {path}: {exc}")


def validate_tensor(item, field, expected_dim, shard_name, record_index):
    if field not in item:
        fail(f"{shard_name} record {record_index} missing {field}")

    value = item[field]
    if not isinstance(value, torch.Tensor):
        fail(f"{shard_name} record {record_index} {field} must be a torch.Tensor")

    shape = tuple(value.squeeze().shape)
    if shape != (expected_dim,):
        fail(
            f"{shard_name} record {record_index} {field} shape {shape} "
            f"does not match ({expected_dim},)"
        )


def validate_shard(bundle_dir, shard_entry, input_dim, output_dim):
    if not isinstance(shard_entry, dict):
        fail("each shard entry must be an object")

    shard_relpath = normalize_shard_path(shard_entry.get("path"))
    shard_path = bundle_dir.joinpath(*shard_relpath.parts)
    if not shard_path.is_file():
        fail(f"listed shard file not found: {shard_relpath}")

    if "sha256" in shard_entry:
        actual_digest = sha256_file(shard_path)
        if actual_digest != shard_entry["sha256"]:
            fail(f"sha256 mismatch for {shard_relpath}")

    data = load_shard(shard_path)
    if not isinstance(data, list) or not data:
        fail(f"{shard_relpath} must load as a non-empty list")

    expected_records = shard_entry.get("records")
    if expected_records is not None:
        if not isinstance(expected_records, int) or expected_records < 0:
            fail(f"{shard_relpath} records must be a non-negative integer")
        if expected_records != len(data):
            fail(
                f"{shard_relpath} records={expected_records} "
                f"does not match loaded count {len(data)}"
            )

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            fail(f"{shard_relpath} record {index} must be a dict")
        validate_tensor(item, "src_vector", input_dim, str(shard_relpath), index)
        validate_tensor(item, "tgt_vector", output_dim, str(shard_relpath), index)

    return str(shard_relpath), len(data)


def validate_bundle(bundle_dir):
    manifest = read_manifest(bundle_dir)
    input_dim = manifest["input_dim"]
    output_dim = manifest["output_dim"]

    shard_reports = [
        validate_shard(bundle_dir, shard_entry, input_dim, output_dim)
        for shard_entry in manifest["shards"]
    ]
    total_records = sum(records for _, records in shard_reports)

    return {
        "bundle_dir": str(bundle_dir),
        "schema_version": manifest["schema_version"],
        "input_dim": input_dim,
        "output_dim": output_dim,
        "shards": shard_reports,
        "total_records": total_records,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Validate an external LIP latent bundle.")
    parser.add_argument("--bundle-dir", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    bundle_dir = args.bundle_dir
    if not bundle_dir.is_dir():
        fail(f"bundle directory does not exist: {bundle_dir}")

    report = validate_bundle(bundle_dir)
    print("Latent bundle validation passed")
    print(f"bundle_dir: {report['bundle_dir']}")
    print(f"schema_version: {report['schema_version']}")
    print(f"dimensions: {report['input_dim']} -> {report['output_dim']}")
    print(f"shards: {len(report['shards'])}")
    print(f"records: {report['total_records']}")
    for shard_name, records in report["shards"]:
        print(f"- {shard_name}: {records} records")


if __name__ == "__main__":
    main()
