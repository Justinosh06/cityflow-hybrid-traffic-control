import json
import math
from pathlib import Path

try:
    from . import evaluate, train
except ImportError:
    import evaluate
    import train


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _project_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def evaluate_pure_rf(
    config_path="data/config.json",
    model_path="models/rf_pretrain.joblib",
    output_csv="data/rf_baseline_metrics.csv",
    max_steps=3600,
    thread_num=4,
    scenario_count=5,
    demand_variation=0.2,
    vehicle_speed_variation=0.15,
    breakdown_probability=1.0,
    breakdown_duration_steps=60,
    minimum_green_duration=20.0,
    decision_interval=10.0,
    maximum_red_time=80.0,
    seed=0,
):
    config_file_path = _project_path(config_path)
    if not config_file_path.is_file():
        raise FileNotFoundError(f"CityFlow config not found: {config_file_path}")

    config = train._load_cityflow_config(config_file_path)
    roadnet_path = Path(config["dir"]) / config["roadnetFile"]
    flow_path = Path(config["dir"]) / config["flowFile"]
    roadnet = json.loads(roadnet_path.read_text(encoding="utf-8"))
    flows = json.loads(flow_path.read_text(encoding="utf-8"))
    intersection_specs = train._build_intersection_specs(roadnet)
    rf_bundle = train._load_rf_bundle(model_path)[0]
    interval = config.get("interval", 1.0)
    if interval <= 0:
        raise ValueError("CityFlow interval must be positive.")
    if minimum_green_duration <= 0 or decision_interval <= 0:
        raise ValueError("Green duration and decision interval must be positive.")
    if maximum_red_time <= 0:
        raise ValueError("maximum_red_time must be positive.")

    decision_steps = max(1, math.ceil(decision_interval / interval))
    minimum_green_steps = max(
        1, math.ceil(minimum_green_duration / interval)
    )
    scenarios = evaluate._build_evaluation_scenarios(
        roadnet,
        scenario_count,
        max_steps,
        demand_variation,
        vehicle_speed_variation,
        breakdown_probability,
        breakdown_duration_steps,
        seed,
    )
    action_fn = evaluate._make_rf_action_fn(
        rf_bundle,
        intersection_specs,
        interval,
        decision_steps,
        minimum_green_steps,
        maximum_red_time,
    )
    metrics = evaluate._run_policy(
        config,
        roadnet_path,
        flows,
        scenarios,
        "random_forest",
        list(intersection_specs),
        max_steps,
        thread_num,
        action_fn,
    )[0]

    output_file_path = _project_path(output_csv)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_file_path, index=False)
    scenario_results = metrics.groupby("scenario").agg(
        final_average_travel_time=("average_travel_time", "last"),
        peak_queue_length=("total_queue_length", "max"),
    )
    print(
        "Pure RF evaluation complete: mean final average travel time "
        f"{scenario_results['final_average_travel_time'].mean():.2f}s; "
        f"mean peak queue "
        f"{scenario_results['peak_queue_length'].mean():.2f} vehicles."
    )
    print(f"Step-by-step RF metrics saved to {output_file_path}")
    return metrics


if __name__ == "__main__":
    evaluate_pure_rf()
