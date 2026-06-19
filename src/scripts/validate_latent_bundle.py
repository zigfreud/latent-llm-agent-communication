import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import torch


REQUIRED_MANIFEST_FIELDS = {
    "bundle_format",
    "schema_version",
    "trace_id",
    "source_model",
    "target_model",
    "dataset_origin",
    "input_dim",
    "output_dim",
    "num_samples",
    "created_at",
    "license_notes",
    "shards",
}
REQUIRED_STRING_FIELDS = {
    "trace_id",
    "source_model",
    "target_model",
    "dataset_origin",
    "created_at",
    "license_notes",
}
OPTIONAL_STRING_FIELDS = {
    "extraction_commit",
    "extraction_notes",
    "source_layer",
    "target_layer",
    "prompt_policy",
}
EXPECTED_BUNDLE_FORMAT = "lip_latent_bundle"
EXPECTED_SCHEMA_VERSION = 1


class BundleValidationError(Exception):
    pass


def fail(message):
    raise BundleValidationError(message)


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

    for field in REQUIRED_STRING_FIELDS:
        if not isinstance(manifest[field], str) or not manifest[field].strip():
            fail(f"{field} must be a non-empty string")

    for field in OPTIONAL_STRING_FIELDS:
        if field in manifest and not isinstance(manifest[field], str):
            fail(f"{field} must be a string when provided")

    for field in ("input_dim", "output_dim", "num_samples"):
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

    if "\\" in path_value:
        fail(f"shard path must use forward slashes: {path_value}")

    shard_path = PurePosixPath(path_value)
    if shard_path.is_absolute() or ".." in shard_path.parts:
        fail(f"shard path must be relative and stay inside the bundle: {path_value}")

    if len(shard_path.parts) != 2 or shard_path.parts[0] != "shards":
        fail(f"shard path must be a direct child of shards/: {path_value}")

    if shard_path.suffix != ".pt":
        fail(f"shard path must end in .pt: {path_value}")

    return shard_path


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_shard(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        fail(
            f"failed to load shard {path} with weights_only=True: {exc}. "
            "Use a PyTorch version that supports weights_only=True and save shards "
            "in a weights_only-compatible tensor/list/dict format."
        )
    except Exception as exc:
        fail(
            f"failed to load shard {path} with weights_only=True: {exc}. "
            "Shards must be saved in a weights_only-compatible tensor/list/dict format."
        )


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

    shard_digest = sha256_file(shard_path)
    if "sha256" in shard_entry:
        if not isinstance(shard_entry["sha256"], str) or not shard_entry["sha256"].strip():
            fail(f"{shard_relpath} sha256 must be a non-empty string when provided")
        if shard_digest != shard_entry["sha256"]:
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

    return {
        "path": str(shard_relpath),
        "records": len(data),
        "manifest_records": expected_records,
        "sha256": shard_digest,
    }


def validate_bundle(bundle_dir):
    manifest = read_manifest(bundle_dir)
    input_dim = manifest["input_dim"]
    output_dim = manifest["output_dim"]
    num_samples = manifest["num_samples"]

    shard_reports = [
        validate_shard(bundle_dir, shard_entry, input_dim, output_dim)
        for shard_entry in manifest["shards"]
    ]
    total_records = sum(shard["records"] for shard in shard_reports)
    if total_records != num_samples:
        fail(f"loaded record count {total_records} does not match num_samples={num_samples}")

    manifest_record_values = [shard["manifest_records"] for shard in shard_reports]
    if any(value is not None for value in manifest_record_values):
        if any(value is None for value in manifest_record_values):
            fail("records must be provided for every shard when any shard provides records")
        manifest_record_total = sum(manifest_record_values)
        if manifest_record_total != num_samples:
            fail(
                f"sum of shard records {manifest_record_total} "
                f"does not match num_samples={num_samples}"
            )

    return {
        "bundle_dir": str(bundle_dir),
        "trace_id": manifest["trace_id"],
        "source_model": manifest["source_model"],
        "target_model": manifest["target_model"],
        "dataset_origin": manifest["dataset_origin"],
        "input_dim": input_dim,
        "output_dim": output_dim,
        "total_records": total_records,
        "shards": shard_reports,
        "validation_status": "passed",
    }


def write_report(report_path, report):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate an external LIP latent bundle.")
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--report-json", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    bundle_dir = args.bundle_dir

    try:
        if not bundle_dir.is_dir():
            fail(f"bundle directory does not exist: {bundle_dir}")

        report = validate_bundle(bundle_dir)
    except BundleValidationError as exc:
        if args.report_json:
            write_report(args.report_json, {
                "bundle_dir": str(bundle_dir),
                "validation_status": "failed",
                "error": str(exc),
            })
        raise SystemExit(f"Invalid latent bundle: {exc}") from exc

    if args.report_json:
        write_report(args.report_json, report)

    print("Latent bundle validation passed")
    print(f"bundle_dir: {report['bundle_dir']}")
    print(f"trace_id: {report['trace_id']}")
    print(f"source_model: {report['source_model']}")
    print(f"target_model: {report['target_model']}")
    print(f"dataset_origin: {report['dataset_origin']}")
    print(f"dimensions: {report['input_dim']} -> {report['output_dim']}")
    print(f"shards: {len(report['shards'])}")
    print(f"records: {report['total_records']}")
    for shard in report["shards"]:
        print(f"- {shard['path']}: {shard['records']} records")


if __name__ == "__main__":
    main()
