"""Content-addressed packet bundles for learned LIP bridge training."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath

import torch
from torch.utils.data import Dataset


PACKET_BUNDLE_FORMAT = "lip_packet_bundle"
PACKET_BUNDLE_SCHEMA_VERSION = 1
PACKET_SPLITS = ("train", "development_selection", "development_gate", "confirmation")
HASH_FIELDS = (
    "task_sha256",
    "prompt_sha256",
    "source_prompt_sha256",
    "target_prompt_sha256",
    "source_input_ids_sha256",
    "source_attention_mask_sha256",
    "target_input_ids_sha256",
    "target_attention_mask_sha256",
)
TORCH_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class PacketBundleValidationError(ValueError):
    pass


def _fail(message: str) -> None:
    raise PacketBundleValidationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def safe_load_packet_shard(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        _fail(f"PyTorch must support weights_only=True to load {path}: {exc}")
    except Exception as exc:
        _fail(f"failed to safely load packet shard {path}: {exc}")


def _read_manifest(bundle_dir: Path) -> dict:
    path = bundle_dir / "manifest.json"
    if not path.is_file():
        _fail(f"missing packet manifest: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _fail(f"packet manifest is not valid JSON: {exc}")
    if not isinstance(manifest, dict):
        _fail("packet manifest must contain a JSON object")
    return manifest


def _validate_endpoint(endpoint, *, label: str) -> None:
    if not isinstance(endpoint, dict):
        _fail(f"{label} must be an object")
    for field in ("model_id", "revision"):
        if not isinstance(endpoint.get(field), str) or not endpoint[field].strip():
            _fail(f"{label}.{field} must be a non-empty string")
    prompt_protocol = endpoint.get("prompt_protocol")
    if not (
        isinstance(prompt_protocol, dict)
        or (isinstance(prompt_protocol, str) and prompt_protocol.strip())
    ):
        _fail(f"{label}.prompt_protocol must be a mapping or non-empty string")


def _validate_dataset(dataset) -> None:
    if not isinstance(dataset, dict):
        _fail("dataset must be an object")
    for field in ("dataset_id", "dataset_config", "revision"):
        if not isinstance(dataset.get(field), str) or not dataset[field].strip():
            _fail(f"dataset.{field} must be a non-empty string")


def _validate_packet_contract(contract, *, label: str) -> tuple[int, int, int]:
    if not isinstance(contract, dict):
        _fail(f"{label} must be an object")
    shape = contract.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or any(not isinstance(value, int) or value <= 0 for value in shape)
    ):
        _fail(f"{label}.shape must contain three positive integers")
    layers, positions, width = shape
    layer_indices = contract.get("layer_indices")
    offsets = contract.get("offsets")
    if not isinstance(layer_indices, list) or len(layer_indices) != layers:
        _fail(f"{label}.layer_indices length must match shape[0]")
    if any(not isinstance(value, int) for value in layer_indices):
        _fail(f"{label}.layer_indices must contain integers")
    if len(layer_indices) != len(set(layer_indices)):
        _fail(f"{label}.layer_indices must be unique")
    if not isinstance(offsets, list) or len(offsets) != positions:
        _fail(f"{label}.offsets length must match shape[1]")
    if any(not isinstance(value, int) for value in offsets):
        _fail(f"{label}.offsets must contain integers")
    if len(offsets) != len(set(offsets)):
        _fail(f"{label}.offsets must be unique")
    if contract.get("state_type") != "residual_input":
        _fail(f"{label}.state_type must be residual_input")
    if contract.get("dtype") not in {"float16", "bfloat16", "float32"}:
        _fail(f"{label}.dtype must be float16, bfloat16, or float32")
    return layers, positions, width


def _normalize_shard_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        _fail("packet shard paths must be non-empty POSIX paths")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        _fail(f"packet shard path escapes the bundle: {value}")
    if len(path.parts) != 2 or path.parts[0] != "shards" or path.suffix != ".pt":
        _fail(f"packet shard must be a direct shards/*.pt file: {value}")
    return path


def _validate_record(
    record,
    *,
    source_shape: tuple[int, int, int],
    target_shape: tuple[int, int, int],
    source_dtype: torch.dtype,
    target_dtype: torch.dtype,
    shard_label: str,
    record_index: int,
) -> tuple[str, str]:
    prefix = f"{shard_label} record {record_index}"
    if not isinstance(record, dict):
        _fail(f"{prefix} must be an object")
    task_id = record.get("task_id")
    split = record.get("split")
    if not isinstance(task_id, str) or not task_id:
        _fail(f"{prefix} task_id must be a non-empty string")
    if split not in PACKET_SPLITS:
        _fail(f"{prefix} split must be one of {', '.join(PACKET_SPLITS)}")
    for field in HASH_FIELDS:
        if not _is_sha256(record.get(field)):
            _fail(f"{prefix} {field} must be a SHA-256 digest")
    name_token_count = record.get("name_token_count")
    if not isinstance(name_token_count, int) or name_token_count <= 0:
        _fail(f"{prefix} name_token_count must be a positive integer")

    for endpoint in ("source", "target"):
        input_ids = record.get(f"{endpoint}_input_ids")
        attention_mask = record.get(f"{endpoint}_attention_mask")
        token_count = record.get(f"{endpoint}_token_count")
        if (
            not isinstance(input_ids, list)
            or not input_ids
            or any(not isinstance(value, int) or value < 0 for value in input_ids)
        ):
            _fail(f"{prefix} {endpoint}_input_ids must contain token IDs")
        if (
            not isinstance(attention_mask, list)
            or len(attention_mask) != len(input_ids)
            or any(value not in (0, 1) for value in attention_mask)
        ):
            _fail(f"{prefix} {endpoint}_attention_mask must align with input IDs")
        if token_count != sum(attention_mask) or token_count <= 0:
            _fail(f"{prefix} {endpoint}_token_count does not match attention mask")
        if sha256_json(input_ids) != record[f"{endpoint}_input_ids_sha256"]:
            _fail(f"{prefix} {endpoint}_input_ids_sha256 does not match")
        if sha256_json(attention_mask) != record[f"{endpoint}_attention_mask_sha256"]:
            _fail(f"{prefix} {endpoint}_attention_mask_sha256 does not match")

    for field, expected, expected_dtype in (
        ("source_packet", source_shape, source_dtype),
        ("target_packet", target_shape, target_dtype),
    ):
        tensor = record.get(field)
        if not isinstance(tensor, torch.Tensor):
            _fail(f"{prefix} {field} must be a tensor")
        if tuple(tensor.shape) != expected:
            _fail(f"{prefix} {field} shape {tuple(tensor.shape)} does not match {expected}")
        if not tensor.dtype.is_floating_point:
            _fail(f"{prefix} {field} must use a floating dtype")
        if tensor.dtype != expected_dtype:
            _fail(
                f"{prefix} {field} dtype {tensor.dtype} does not match "
                f"{expected_dtype}"
            )
        if not bool(torch.isfinite(tensor).all()):
            _fail(f"{prefix} {field} contains non-finite values")
    return split, task_id


def validate_packet_bundle(
    bundle_dir: Path | str,
    *,
    require_real: bool = False,
) -> dict:
    """Validate manifest, provenance, shard set, record layouts, and split counts."""

    bundle_dir = Path(bundle_dir)
    if not bundle_dir.is_dir():
        _fail(f"packet bundle directory does not exist: {bundle_dir}")
    manifest = _read_manifest(bundle_dir)
    if manifest.get("bundle_format") != PACKET_BUNDLE_FORMAT:
        _fail(f"bundle_format must be {PACKET_BUNDLE_FORMAT}")
    if manifest.get("schema_version") != PACKET_BUNDLE_SCHEMA_VERSION:
        _fail(f"schema_version must be {PACKET_BUNDLE_SCHEMA_VERSION}")
    if not isinstance(manifest.get("trace_id"), str) or not manifest["trace_id"]:
        _fail("trace_id must be a non-empty string")
    extraction_mode = manifest.get("extraction_mode")
    if extraction_mode not in {"real", "dry_run"}:
        _fail("extraction_mode must be real or dry_run")
    if require_real and extraction_mode != "real":
        _fail("claim-oriented packet training requires extraction_mode=real")
    _validate_endpoint(manifest.get("source"), label="source")
    _validate_endpoint(manifest.get("target"), label="target")
    _validate_dataset(manifest.get("dataset"))
    if not _is_sha256(manifest.get("config_sha256")):
        _fail("config_sha256 must be a SHA-256 digest")
    registry = manifest.get("registry")
    if not isinstance(registry, dict):
        _fail("registry must bind the task registry manifest")
    if not _is_sha256(registry.get("manifest_sha256")):
        _fail("registry.manifest_sha256 must be a SHA-256 digest")
    source_shape = _validate_packet_contract(
        manifest.get("source_packet"), label="source_packet"
    )
    target_shape = _validate_packet_contract(
        manifest.get("target_packet"), label="target_packet"
    )
    source_dtype = TORCH_DTYPES[manifest["source_packet"]["dtype"]]
    target_dtype = TORCH_DTYPES[manifest["target_packet"]["dtype"]]
    predecessor = manifest.get("predecessor")
    if not isinstance(predecessor, dict):
        _fail("predecessor must bind the packet bundle to LIP-PROTO-013")
    if predecessor.get("protocol") != "LIP-PROTO-013":
        _fail("predecessor.protocol must be LIP-PROTO-013")
    if not _is_sha256(predecessor.get("sha256sums_sha256")):
        _fail("predecessor.sha256sums_sha256 must be a SHA-256 digest")

    split_counts = manifest.get("splits")
    if not isinstance(split_counts, dict) or set(split_counts) != set(PACKET_SPLITS):
        _fail("splits must define train, development_selection, development_gate, confirmation")
    if any(not isinstance(value, int) or value < 0 for value in split_counts.values()):
        _fail("split counts must be non-negative integers")

    shard_entries = manifest.get("shards")
    if not isinstance(shard_entries, list) or not shard_entries:
        _fail("shards must be a non-empty list")
    listed_paths = []
    records = []
    for entry in shard_entries:
        if not isinstance(entry, dict):
            _fail("each packet shard entry must be an object")
        relative = _normalize_shard_path(entry.get("path"))
        relative_text = relative.as_posix()
        if relative_text in listed_paths:
            _fail(f"duplicate packet shard path: {relative_text}")
        listed_paths.append(relative_text)
        path = bundle_dir.joinpath(*relative.parts)
        if not path.is_file():
            _fail(f"listed packet shard does not exist: {relative_text}")
        if not _is_sha256(entry.get("sha256")):
            _fail(f"{relative_text} sha256 must be a SHA-256 digest")
        if sha256_file(path) != entry["sha256"]:
            _fail(f"sha256 mismatch for {relative_text}")
        shard_records = safe_load_packet_shard(path)
        if not isinstance(shard_records, list) or not shard_records:
            _fail(f"{relative_text} must contain a non-empty record list")
        if entry.get("records") != len(shard_records):
            _fail(f"{relative_text} record count does not match the manifest")
        for index, record in enumerate(shard_records):
            _validate_record(
                record,
                source_shape=source_shape,
                target_shape=target_shape,
                source_dtype=source_dtype,
                target_dtype=target_dtype,
                shard_label=relative_text,
                record_index=index,
            )
        records.extend(shard_records)

    shard_dir = bundle_dir / "shards"
    actual_paths = sorted(
        f"shards/{path.name}" for path in shard_dir.glob("*.pt") if path.is_file()
    )
    nested = [path for path in shard_dir.rglob("*.pt") if path.parent != shard_dir]
    if nested:
        _fail("nested packet shards are not allowed")
    if sorted(listed_paths) != actual_paths:
        _fail("listed and actual packet shard sets differ")

    task_keys = [(record["split"], record["task_id"]) for record in records]
    if len(task_keys) != len(set(task_keys)):
        _fail("packet records contain duplicate split/task IDs")
    observed_splits = Counter(record["split"] for record in records)
    expected_nonzero_splits = {
        name: split_counts[name]
        for name in PACKET_SPLITS
        if split_counts[name]
    }
    if dict(observed_splits) != expected_nonzero_splits:
        for split in PACKET_SPLITS:
            if observed_splits.get(split, 0) != split_counts[split]:
                _fail(f"observed {split} count does not match the manifest")
    if manifest.get("task_keys_sha256") != sha256_json(task_keys):
        _fail("task_keys_sha256 does not match packet record order")

    return {
        "bundle_dir": str(bundle_dir),
        "trace_id": manifest["trace_id"],
        "extraction_mode": extraction_mode,
        "source_shape": list(source_shape),
        "target_shape": list(target_shape),
        "split_counts": {name: observed_splits.get(name, 0) for name in PACKET_SPLITS},
        "records": len(records),
        "shards": listed_paths,
        "validation_status": "passed",
    }


def load_packet_records(bundle_dir: Path | str, *, split: str | None = None) -> list[dict]:
    bundle_dir = Path(bundle_dir)
    manifest = _read_manifest(bundle_dir)
    records = []
    for entry in manifest.get("shards", []):
        relative = _normalize_shard_path(entry.get("path"))
        records.extend(safe_load_packet_shard(bundle_dir.joinpath(*relative.parts)))
    if split is not None:
        if split not in PACKET_SPLITS:
            raise ValueError(f"unknown packet split: {split}")
        records = [record for record in records if record["split"] == split]
    return records


class PacketRecordDataset(Dataset):
    """In-memory task-level packet dataset with training-only normalization."""

    def __init__(
        self,
        records: Sequence[Mapping],
        *,
        scaffold: torch.Tensor,
        site_scale: torch.Tensor,
    ) -> None:
        if not records:
            raise ValueError("packet dataset requires at least one record")
        self.records = list(records)
        self.scaffold = scaffold.float()
        self.site_scale = site_scale.float()
        if self.scaffold.ndim != 3 or self.site_scale.shape != self.scaffold.shape[:2]:
            raise ValueError("scaffold/site_scale shapes are inconsistent")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        target = record["target_packet"].float()
        normalized = (target - self.scaffold) / self.site_scale[:, :, None]
        return {
            "task_id": record["task_id"],
            "source_packet": record["source_packet"].float(),
            "target_residual": normalized,
            "name_token_count": int(record["name_token_count"]),
        }


def compute_target_packet_statistics(
    records: Iterable[Mapping],
    *,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute training-only target scaffold and scalar RMS for every receiver site."""

    count = 0
    packet_sum = None
    packet_square_sum = None
    expected_shape = None
    for record in records:
        packet = record["target_packet"].detach().to(dtype=torch.float64, device="cpu")
        if packet.ndim != 3:
            raise ValueError("target_packet must have [layers, positions, width]")
        if expected_shape is None:
            expected_shape = tuple(packet.shape)
            packet_sum = torch.zeros_like(packet)
            packet_square_sum = torch.zeros_like(packet)
        elif tuple(packet.shape) != expected_shape:
            raise ValueError("all target packets must share one shape")
        packet_sum += packet
        packet_square_sum += packet.square()
        count += 1
    if count == 0:
        raise ValueError("cannot compute packet statistics from zero records")
    scaffold = packet_sum / count
    element_variance = (packet_square_sum / count - scaffold.square()).clamp_min(0.0)
    site_scale = torch.sqrt(element_variance.mean(dim=-1) + float(epsilon))
    return scaffold.float(), site_scale.float()
