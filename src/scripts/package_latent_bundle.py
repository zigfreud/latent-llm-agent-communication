import argparse
import json
import zipfile
from pathlib import Path, PurePosixPath


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
EXPECTED_BUNDLE_FORMAT = "lip_latent_bundle"
EXPECTED_SCHEMA_VERSION = 1
BLOCKED_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "checkpoints",
    "models",
    "runs",
}
BLOCKED_SUFFIXES = {".bin", ".gguf", ".pth"}


def fail(message):
    raise SystemExit(f"Cannot package latent bundle: {message}")


def read_manifest(manifest_path):
    if not manifest_path.is_file():
        fail(f"manifest file does not exist: {manifest_path}")

    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except json.JSONDecodeError as exc:
        fail(f"manifest is not valid JSON: {exc}")

    if not isinstance(manifest, dict):
        fail("manifest must contain a JSON object")

    missing = sorted(REQUIRED_MANIFEST_FIELDS.difference(manifest))
    if missing:
        fail(f"manifest missing required field(s): {', '.join(missing)}")

    if manifest["bundle_format"] != EXPECTED_BUNDLE_FORMAT:
        fail(f"bundle_format must be {EXPECTED_BUNDLE_FORMAT}")

    if manifest["schema_version"] != EXPECTED_SCHEMA_VERSION:
        fail(f"schema_version must be {EXPECTED_SCHEMA_VERSION}")

    for field in REQUIRED_STRING_FIELDS:
        if not isinstance(manifest[field], str) or not manifest[field].strip():
            fail(f"{field} must be a non-empty string")

    for field in ("input_dim", "output_dim", "num_samples"):
        if not isinstance(manifest[field], int) or manifest[field] <= 0:
            fail(f"{field} must be a positive integer")

    shards = manifest["shards"]
    if not isinstance(shards, list) or not shards:
        fail("manifest must include a non-empty shards list")

    if not any(
        isinstance(shard, dict) and shard.get("path") == "shards/shard_0.pt"
        for shard in shards
    ):
        fail("manifest must list required shard path: shards/shard_0.pt")

    if any(isinstance(shard, dict) and "records" in shard for shard in shards):
        if any(not isinstance(shard, dict) or "records" not in shard for shard in shards):
            fail("records must be provided for every shard when any shard provides records")
        for shard in shards:
            if not isinstance(shard["records"], int) or shard["records"] < 0:
                fail("records must be a non-negative integer for every shard")
        records_total = sum(shard["records"] for shard in shards)
        if records_total != manifest["num_samples"]:
            fail(
                f"sum of shard records {records_total} "
                f"does not match num_samples={manifest['num_samples']}"
            )

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

    if any(part in BLOCKED_PARTS for part in shard_path.parts):
        fail(f"shard path includes a blocked directory: {path_value}")

    if shard_path.suffix in BLOCKED_SUFFIXES:
        fail(f"blocked file type in shard path: {path_value}")

    return shard_path


def shard_source_path(shards_dir, shard_relpath):
    return shards_dir.joinpath(*shard_relpath.parts[1:])


def package_bundle(shards_dir, manifest_path, output_zip):
    if not shards_dir.is_dir():
        fail(f"shards directory does not exist: {shards_dir}")

    manifest = read_manifest(manifest_path)
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(manifest_path, "manifest.json")

        for shard_entry in manifest["shards"]:
            if not isinstance(shard_entry, dict):
                fail("each shard entry must be an object")

            shard_relpath = normalize_shard_path(shard_entry.get("path"))
            if shard_relpath in seen:
                fail(f"duplicate shard path: {shard_relpath}")
            seen.add(shard_relpath)

            source_path = shard_source_path(shards_dir, shard_relpath)
            if not source_path.is_file():
                fail(f"listed shard file not found in shards-dir: {source_path}")

            archive.write(source_path, shard_relpath.as_posix())

    return output_zip


def parse_args():
    parser = argparse.ArgumentParser(description="Package a LIP latent bundle zip.")
    parser.add_argument("--shards-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-zip", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    output_zip = package_bundle(args.shards_dir, args.manifest, args.output_zip)
    print(f"Wrote {output_zip}")


if __name__ == "__main__":
    main()
