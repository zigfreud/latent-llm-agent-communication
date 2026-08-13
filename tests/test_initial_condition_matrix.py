from src.pipelines.initial_condition_matrix import summarize_initial_condition_matrix


def _metrics(rmse, retrieval):
    return {
        "normalized_residual_rmse": rmse,
        "regions": {
            region: {"retrieval_top1": retrieval}
            for region in ("joint", "core", "name")
        },
    }


def _holm(passed):
    return {"passed": passed, "family": []}


def test_matrix_gate_requires_primary_replicas_and_paired_improvement():
    cells = []
    for seed, primary_passed in ((4001, True), (4003, True), (4007, False)):
        cells.extend(
            [
                {
                    "variant": "static_entry_snapshot",
                    "seed": seed,
                    "development_gate": _holm(False),
                    "development_gate_metrics": {
                        "induced_trajectory": _metrics(1.0, 0.25)
                    },
                },
                {
                    "variant": "unrolled_initial_condition",
                    "seed": seed,
                    "development_gate": _holm(primary_passed),
                    "development_gate_metrics": {
                        "induced_trajectory": _metrics(
                            0.8 if seed != 4007 else 1.1,
                            0.25,
                        )
                    },
                },
            ]
        )
    experiment = {
        "training": {"seeds": [4001, 4003, 4007]},
        "development_gate": {
            "minimum_passing_primary_replicas": 2,
            "pass_action": "confirm",
            "fail_action": "redesign",
        },
    }

    summary = summarize_initial_condition_matrix(cells, experiment)

    assert summary["primary_multi_replica_gate"]["passed"]
    assert summary["paired_primary_vs_static"]["passing_pairs"] == 2
    assert summary["paired_primary_vs_static"]["passed"]
    assert summary["passed"]
