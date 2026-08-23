from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
import yaml

from meridian.adapters.surrogate import BoundarySurrogateAdapter
from meridian.ingest import ingest_libero_rollouts
from meridian.intervention import (
    evaluate_checkpoint,
    make_intervention_arms,
    materialize_surrogate_dataset,
)
from meridian.models import ExperimentSpec, InterventionArm
from meridian.report import write_research_report
from meridian.scientist import evidence_package, propose_competing_hypotheses
from meridian.search import AdaptiveFailureSearch, build_capability_map, propose_parameter_points
from meridian.store import ExperimentStore

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Meridian policy intervention harness."""


@app.command("surrogate-e2e")
def surrogate_e2e(
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/surrogate.yaml"),
    output: Annotated[Path, typer.Option()] = Path("artifacts/surrogate-e2e"),
) -> None:
    """Exercise the full scientific contract without claiming a robotics result."""
    spec = ExperimentSpec.model_validate(yaml.safe_load(config.read_text()))
    output.mkdir(parents=True, exist_ok=True)
    with ExperimentStore(output / "experiment.sqlite") as store:
        store.put("experiment", spec)
        adapter = BoundarySurrogateAdapter()
        adapter.load(spec.checkpoint_id, spec.checkpoint_source)
        rollouts = AdaptiveFailureSearch(spec, adapter, store).run(seed=17)
        capability_map = build_capability_map(spec, rollouts)
        store.put("capability_map", capability_map)
        evidence_package(spec, capability_map, rollouts, output / "scientist_evidence.json")
        hypotheses = propose_competing_hypotheses(spec, capability_map, rollouts)
        hypotheses[0].status = "selected"
        for hypothesis in hypotheses:
            store.put("hypothesis", hypothesis)
        arms = make_intervention_arms(
            spec, hypotheses[0], capability_map, dose=24, training_steps=100
        )
        evaluations = []
        for arm in arms:
            store.put("intervention", arm)
            dataset = materialize_surrogate_dataset(spec, arm, output / "datasets")
            store.put("dataset", dataset)
            variant = BoundarySurrogateAdapter()
            variant.load(spec.checkpoint_id, spec.checkpoint_source)
            checkpoint = spec.checkpoint_id
            if arm.arm != InterventionArm.NONE:
                checkpoint, cost = variant.finetune(
                    dataset, arm.id, arm.training_steps, arm.seed, output / "checkpoints"
                )
                store.event(
                    spec.id, "training.completed", {"cost": cost.model_dump(mode="json")}, arm.id
                )
            result = evaluate_checkpoint(variant, spec, checkpoint, arm)
            store.put("evaluation", result)
            evaluations.append(result)
        report = write_research_report(
            spec, capability_map, hypotheses, evaluations, output / "RESULT.md"
        )
        typer.echo(f"experiment={spec.id} rollouts={len(rollouts)} report={report}")


@app.command("map-results")
def map_results(
    config: Annotated[Path, typer.Option(exists=True)],
    rollouts: Annotated[Path, typer.Option(exists=True)],
    output: Annotated[Path, typer.Option()],
) -> None:
    """Ingest parameterized LIBERO results and produce a capability/scientist packet."""
    spec = ExperimentSpec.model_validate(yaml.safe_load(config.read_text()))
    records = ingest_libero_rollouts(spec, rollouts)
    capability_map = build_capability_map(spec, records)
    output.mkdir(parents=True, exist_ok=True)
    with ExperimentStore(output / "experiment.sqlite") as store:
        store.put("experiment", spec)
        for record in records:
            store.put("rollout", record)
        store.put("capability_map", capability_map)
    map_path = output / "capability_map.json"
    map_path.write_text(
        json.dumps(capability_map.model_dump(mode="json"), indent=2, sort_keys=True)
    )
    packet = evidence_package(spec, capability_map, records, output / "scientist_evidence.json")
    typer.echo(f"experiment={spec.id} rollouts={len(records)} map={map_path} evidence={packet}")


@app.command("plan-search")
def plan_search(
    config: Annotated[Path, typer.Option(exists=True)],
    output: Annotated[Path, typer.Option()],
    task_id: Annotated[int, typer.Option(min=0)] = 0,
    count: Annotated[int, typer.Option(min=1)] = 24,
    seed: Annotated[int, typer.Option()] = 1000,
    prior_rollouts: Annotated[Path | None, typer.Option(exists=True)] = None,
) -> None:
    """Propose an initial or boundary-adaptive external LIBERO rollout batch."""
    spec = ExperimentSpec.model_validate(yaml.safe_load(config.read_text()))
    prior = ingest_libero_rollouts(spec, prior_rollouts) if prior_rollouts else []
    points = propose_parameter_points(spec, prior, count, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        for index, point in enumerate(points):
            plan = {
                "id": f"{spec.id}-batch{seed}-episode{index:04d}",
                "task_suite": spec.task_suite,
                "task_id": task_id,
                "seed": seed + index,
                **point,
            }
            stream.write(json.dumps(plan, sort_keys=True) + "\n")
    typer.echo(f"experiment={spec.id} prior={len(prior)} proposed={len(points)} plans={output}")


if __name__ == "__main__":
    app()
