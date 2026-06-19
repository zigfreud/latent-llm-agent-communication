import argparse
import json
import zipfile
from pathlib import Path, PurePosixPath


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

    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        fail("manifest must include a non-empty shards list")

    if not any(
        isinstance(shard, dict) and shard.get("path") == "shards/shard_0.pt"
        for shard in shards
    ):
        fail("manifest must list required shard path: shards/shard_0.pt")

    return manifest


def normalize_shard_path(path_value):
    if not isinstance(path_value, str) or not path_value:
        fail("each shard entry must include a non-empty string path")

    shard_path = PurePosixPath(path_value)
    if shard_path.is_absolute() or ".." in shard_path.parts:
        fail(f"shard path must be relative and stay inside the bundle: {path_value}")

    if len(shard_path.parts) < 2 or shard_path.parts[0] != "shards":
        fail(f"shard path must be under shards/: {path_value}")

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
