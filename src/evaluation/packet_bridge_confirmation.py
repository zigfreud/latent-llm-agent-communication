"""Frozen design helpers for the LIP-PROTO-014 functional confirmation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

import torch

from src.evaluation.oracle_terminal_factorial import validate_terminal_layout
from src.evaluation.source_only import derangement_indices


PACKET_CONFIRMATION_EXPERIMENT_ID = "LIP-PROTO-014"
PACKET_CONFIRMATION_PROTOCOL_VERSION = (
    "lip-source-conditioned-residual-packet-v1"
)
PACKET_CONFIRMATION_TASK_COUNT = 32
PACKET_CONFIRMATION_STRATA = (2, 3)
PACKET_CONFIRMATION_TASKS_PER_STRATUM = 16
PACKET_CONFIRMATION_CONDITIONS = (
    "neutral_no_lip",
    "text_only_no_lip",
    "oracle_teacher_matched",
    "oracle_teacher_shuffled",
    "mean_scaffold",
    "learned_matched",
    "learned_shuffled",
    "random_residual_norm_matched",
)
PACKET_CONFIRMATION_SHARED_CONDITIONS = PACKET_CONFIRMATION_CONDITIONS[:5]
PACKET_CONFIRMATION_REPLICA_CONDITIONS = PACKET_CONFIRMATION_CONDITIONS[5:]
PACKET_CONFIRMATION_GENERATION_SEEDS = (4127, 4241, 4357)
PACKET_CONFIRMATION_TRAINING_SEEDS = (4001, 4003, 4007)
PACKET_CONFIRMATION_DERANGEMENT_SEED = 4513
PACKET_CONFIRMATION_STATISTICS_SEED = 4481

# The 014 explicitly inherits the hardened namespace used by its predecessor.
# These are the unchanged LIP-PROTO-013 execution and uncertainty budgets.
PACKET_CONFIRMATION_EVALUATION_POLICY = {
    "timeout_seconds": 5.0,
    "memory_mb": 512,
    "confidence": 0.95,
    "bootstrap_iterations": 10_000,
    "alpha": 0.05,
    "alternative": "greater",
}


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_packet_confirmation_contract(config: Mapping) -> None:
    """Reject any drift in the already-registered confirmation design."""

    confirmation = config.get("confirmation", {})
    checks = {
        "experiment_id": config.get("experiment_id")
        == PACKET_CONFIRMATION_EXPERIMENT_ID,
        "protocol_version": config.get("protocol_version")
        == PACKET_CONFIRMATION_PROTOCOL_VERSION,
        "task_count": confirmation.get("task_count")
        == PACKET_CONFIRMATION_TASK_COUNT,
        "conditions": tuple(confirmation.get("required_conditions", ()))
        == PACKET_CONFIRMATION_CONDITIONS,
        "generation_seeds": tuple(confirmation.get("generation_seeds", ()))
        == PACKET_CONFIRMATION_GENERATION_SEEDS,
        "training_seeds": tuple(config.get("training", {}).get("seeds", ()))
        == PACKET_CONFIRMATION_TRAINING_SEEDS,
        "derangement_seed": confirmation.get("derangement_seed")
        == PACKET_CONFIRMATION_DERANGEMENT_SEED,
        "statistics_seed": confirmation.get("statistics_seed")
        == PACKET_CONFIRMATION_STATISTICS_SEED,
        "rank_slice": confirmation.get("rank_slice_within_capable_stratum")
        == [16, 32],
        "tokenizer_strata": confirmation.get("tokenizer_strata")
        == {2: 16, 3: 16},
        "max_new_tokens": confirmation.get("max_new_tokens") == 256,
        "do_sample": confirmation.get("do_sample") is True,
        "temperature": float(confirmation.get("temperature", -1.0)) == 0.2,
        "top_p": float(confirmation.get("top_p", -1.0)) == 0.95,
        "repetition_penalty": float(
            confirmation.get("repetition_penalty", -1.0)
        )
        == 1.0,
        "development_alpha": float(
            config.get("development_gate", {}).get("alpha", -1.0)
        )
        == PACKET_CONFIRMATION_EVALUATION_POLICY["alpha"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "LIP-PROTO-014 confirmation contract drifted: " + ", ".join(failed)
        )


def packet_confirmation_design_fingerprint(config: Mapping) -> str:
    """Bind generation/evaluation to every frozen 014 design field."""

    validate_packet_confirmation_contract(config)
    return _canonical_sha256(config)


def stratified_confirmation_donors(
    tasks: Sequence[Mapping],
    *,
    seed: int = PACKET_CONFIRMATION_DERANGEMENT_SEED,
) -> dict[int, int]:
    """Return a deterministic same-tokenizer-stratum donor for every task."""

    if len(tasks) != PACKET_CONFIRMATION_TASK_COUNT:
        raise ValueError("confirmation donor plan requires exactly 32 tasks")
    task_ids = [str(task.get("task_id", "")) for task in tasks]
    if any(not task_id for task_id in task_ids) or len(set(task_ids)) != len(
        task_ids
    ):
        raise ValueError("confirmation task IDs must be non-empty and unique")
    strata: dict[int, list[int]] = {count: [] for count in PACKET_CONFIRMATION_STRATA}
    for index, task in enumerate(tasks):
        count = validate_terminal_layout(task.get("terminal_layout", {}))
        strata[count].append(index)
    if any(
        len(indices) != PACKET_CONFIRMATION_TASKS_PER_STRATUM
        for indices in strata.values()
    ):
        raise ValueError("confirmation tasks must contain balanced 16-task strata")

    donors: dict[int, int] = {}
    for count, indices in strata.items():
        permutation = derangement_indices(len(indices), int(seed) + count)
        donors.update(
            {
                target_index: indices[permutation[local_index]]
                for local_index, target_index in enumerate(indices)
            }
        )
    if any(target == donor for target, donor in donors.items()):
        raise RuntimeError("confirmation donor mapping is not a derangement")
    return donors


def expected_confirmation_generation_keys(
    task_ids: Sequence[str],
    *,
    generation_seeds: Sequence[int] = PACKET_CONFIRMATION_GENERATION_SEEDS,
    training_seeds: Sequence[int] = PACKET_CONFIRMATION_TRAINING_SEEDS,
) -> set[tuple[str, str, int, int | None]]:
    """Build the intentionally non-duplicated confirmation generation grid."""

    normalized_ids = [str(task_id) for task_id in task_ids]
    if len(normalized_ids) != PACKET_CONFIRMATION_TASK_COUNT or len(
        set(normalized_ids)
    ) != len(normalized_ids):
        raise ValueError("confirmation grid requires 32 unique task IDs")
    normalized_generation_seeds = [int(seed) for seed in generation_seeds]
    normalized_training_seeds = [int(seed) for seed in training_seeds]
    if tuple(normalized_generation_seeds) != PACKET_CONFIRMATION_GENERATION_SEEDS:
        raise ValueError("confirmation generation seeds changed")
    if tuple(normalized_training_seeds) != PACKET_CONFIRMATION_TRAINING_SEEDS:
        raise ValueError("confirmation bridge replica seeds changed")

    keys: set[tuple[str, str, int, int | None]] = set()
    for task_id in normalized_ids:
        for generation_seed in normalized_generation_seeds:
            keys.update(
                (task_id, condition, generation_seed, None)
                for condition in PACKET_CONFIRMATION_SHARED_CONDITIONS
            )
            keys.update(
                (task_id, condition, generation_seed, training_seed)
                for training_seed in normalized_training_seeds
                for condition in PACKET_CONFIRMATION_REPLICA_CONDITIONS
            )
    return keys


def isotropic_residual_with_matched_layer_norms(
    reference_residual: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Draw isotropic noise and match each receiver layer's residual norm."""

    if not isinstance(reference_residual, torch.Tensor) or reference_residual.ndim != 3:
        raise ValueError("reference_residual must have [layers, positions, width]")
    reference = reference_residual.detach().float().cpu()
    if not bool(torch.isfinite(reference).all()):
        raise ValueError("reference_residual contains non-finite values")
    noise = torch.randn(
        reference.shape,
        dtype=torch.float32,
        device="cpu",
        generator=generator,
    )
    reference_norms = torch.linalg.vector_norm(reference.flatten(1), dim=1)
    noise_norms = torch.linalg.vector_norm(noise.flatten(1), dim=1)
    if not bool(torch.all(noise_norms > 0.0)):
        raise RuntimeError("isotropic residual draw produced a zero-norm layer")
    scaled = noise * (reference_norms / noise_norms)[:, None, None]
    if not bool(torch.isfinite(scaled).all()):
        raise FloatingPointError("norm-matched residual contains non-finite values")
    return scaled


def packet_layer_norms(packet: torch.Tensor) -> list[float]:
    if not isinstance(packet, torch.Tensor) or packet.ndim != 3:
        raise ValueError("packet must have [layers, positions, width]")
    values = torch.linalg.vector_norm(packet.detach().float().cpu().flatten(1), dim=1)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("packet layer norms are non-finite")
    return [float(value) for value in values.tolist()]
