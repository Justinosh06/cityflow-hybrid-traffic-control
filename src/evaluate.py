import json
import math
import os
import random
import shutil
import tempfile
from pathlib import Path

import cityflow
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix

try:
    from . import train
except ImportError:
    import train


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _project_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_cityflow_config(config_path):
    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    resource_dir = Path(config.get("dir", ""))
    if not resource_dir.is_absolute():
        resource_dir = PROJECT_ROOT / resource_dir
    config["dir"] = str(resource_dir.resolve()) + os.sep
    config["saveReplay"] = False
    return config


def _build_evaluation_scenarios(
    roadnet,
    scenario_count,
    max_steps,
    demand_variation,
    vehicle_speed_variation,
    breakdown_probability,
    breakdown_duration_steps,
    seed,
):
    if scenario_count <= 0:
        raise ValueError("scenario_count must be a positive integer.")

    rng = random.Random(seed)
    return [
        {
            "scenario": scenario_index,
            **train._generate_scenario_uncertainties(
                roadnet,
                max_steps,
                rng,
                demand_variation,
                vehicle_speed_variation,
                breakdown_probability,
                breakdown_duration_steps,
            ),
        }
        for scenario_index in range(1, scenario_count + 1)
    ]


def _run_policy(
    config,
    roadnet_path,
    flows,
    scenarios,
    policy_name,
    intersections,
    max_steps,
    thread_num,
    action_fn,
):
    rows = []
    actions_by_step = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for scenario in scenarios:
            scenario_dir = Path(temp_dir) / f"scenario_{scenario['scenario']}"
            scenario_dir.mkdir()
            shutil.copy2(roadnet_path, scenario_dir / "roadnet.json")
            with (scenario_dir / "flow.json").open(
                "w", encoding="utf-8"
            ) as flow_file:
                json.dump(
                    train._scale_flow_demand(
                        flows,
                        scenario["demand_scale"],
                        scenario["vehicle_speed_scale"],
                    ),
                    flow_file,
                )

            scenario_config = dict(config)
            scenario_config.update(
                {
                    "dir": str(scenario_dir.resolve()) + os.sep,
                    "roadnetFile": "roadnet.json",
                    "flowFile": "flow.json",
                    "seed": scenario["seed"],
                    "rlTrafficLight": action_fn is not None,
                }
            )
            engine_config_path = scenario_dir / "config.json"
            with engine_config_path.open("w", encoding="utf-8") as config_file:
                json.dump(scenario_config, config_file)

            engine = cityflow.Engine(
                str(engine_config_path), thread_num=thread_num
            )
            affected_vehicles = {}
            for step in range(1, max_steps + 1):
                train._apply_scenario_incidents(
                    engine,
                    scenario["incidents"],
                    step,
                    affected_vehicles,
                )
                step_actions = {}
                if action_fn is not None:
                    action_observations = action_fn(engine, step, scenario)
                    for intersection_id in intersections:
                        action = action_observations[intersection_id]
                        engine.set_tl_phase(intersection_id, action)
                        step_actions[intersection_id] = action

                engine.next_step()
                lane_waiting_counts = engine.get_lane_waiting_vehicle_count()
                rows.append(
                    {
                        "policy": policy_name,
                        "scenario": scenario["scenario"],
                        "demand_scale": scenario["demand_scale"],
                        "vehicle_speed_scale": scenario["vehicle_speed_scale"],
                        "breakdown_count": len(scenario["incidents"]),
                        "step": step,
                        "total_queue_length": sum(lane_waiting_counts.values()),
                        "average_travel_time": engine.get_average_travel_time(),
                    }
                )
                actions_by_step.append(step_actions)

    return pd.DataFrame(rows), actions_by_step


def _build_action_confusion_matrix(
    rf_actions, q_actions, intersections, known_actions
):
    reference_actions = []
    q_learning_actions = []
    for rf_step, q_step in zip(rf_actions, q_actions):
        for intersection_id in intersections:
            reference_actions.append(rf_step[intersection_id])
            q_learning_actions.append(q_step[intersection_id])

    labels = sorted(
        set(known_actions) | set(reference_actions) | set(q_learning_actions)
    )
    matrix = confusion_matrix(reference_actions, q_learning_actions, labels=labels)
    return pd.DataFrame(
        matrix,
        index=pd.Index(labels, name="random_forest_phase"),
        columns=pd.Index(labels, name="q_learning_phase"),
    )


def run_evaluation(
    config_path="data/config.json",
    model_path="models/rf_pretrain.joblib",
    q_table_path="models/q_table.joblib",
    output_csv="data/evaluation_metrics.csv",
    confusion_matrix_csv="data/evaluation_confusion_matrix.csv",
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
    if max_steps <= 0:
        raise ValueError("max_steps must be a positive integer.")
    if minimum_green_duration <= 0:
        raise ValueError("minimum_green_duration must be positive.")
    if decision_interval <= 0:
        raise ValueError("decision_interval must be positive.")
    if maximum_red_time <= 0:
        raise ValueError("maximum_red_time must be positive.")

    config_file_path = _project_path(config_path)
    model_file_path = _project_path(model_path)
    q_table_file_path = _project_path(q_table_path)
    output_file_path = _project_path(output_csv)
    confusion_matrix_path = _project_path(confusion_matrix_csv)
    if not config_file_path.is_file():
        raise FileNotFoundError(f"CityFlow config not found: {config_file_path}")
    if not model_file_path.is_file():
        raise FileNotFoundError(
            f"Pre-trained Random Forest model not found: {model_file_path}"
        )
    if not q_table_file_path.is_file():
        raise FileNotFoundError(
            f"Q-learning table not found: {q_table_file_path}. "
            "Run train.py first."
        )

    config = _load_cityflow_config(config_file_path)
    roadnet_path = Path(config["dir"]) / config["roadnetFile"]
    with roadnet_path.open("r", encoding="utf-8") as roadnet_file:
        roadnet = json.load(roadnet_file)
    flow_path = Path(config["dir"]) / config["flowFile"]
    with flow_path.open("r", encoding="utf-8") as flow_file:
        flows = json.load(flow_file)
    scenarios = _build_evaluation_scenarios(
        roadnet,
        scenario_count,
        max_steps,
        demand_variation,
        vehicle_speed_variation,
        breakdown_probability,
        breakdown_duration_steps,
        seed,
    )
    intersection_specs = train._build_intersection_specs(roadnet)
    intersections = list(intersection_specs)

    model = joblib.load(model_file_path)
    expected_features = getattr(model, "feature_names_in_", None)
    if expected_features is None:
        raise ValueError(
            "The pre-trained Random Forest does not expose feature_names_in_; "
            "cannot safely align lane-count features."
        )
    expected_features = list(expected_features)
    known_actions = set(getattr(model, "classes_", []))
    known_actions.update(
        action
        for spec in intersection_specs.values()
        for action in spec["actions"]
    )

    def rf_action_fn(engine, _step, _scenario):
        lane_vehicle_counts = engine.get_lane_vehicle_count()
        state = pd.DataFrame(
            [
                {
                    feature: lane_vehicle_counts.get(feature, 0)
                    for feature in expected_features
                }
            ],
            columns=expected_features,
        )
        action = int(model.predict(state)[0])
        return {intersection_id: action for intersection_id in intersections}

    interval = config.get("interval", 1.0)
    if interval <= 0:
        raise ValueError("CityFlow interval must be positive.")
    minimum_green_steps = max(
        1, math.ceil(minimum_green_duration / interval)
    )
    decision_steps = max(1, math.ceil(decision_interval / interval))
    expected_training_settings = train._build_training_settings(
        interval,
        minimum_green_duration,
        decision_interval,
        maximum_red_time,
        longest_wait_weight=0.1,
    )
    q_table = train._load_q_table(
        q_table_file_path,
        intersection_specs,
        expected_training_settings,
    )
    controller_state = {}
    controller_rng = random.Random(seed)

    def q_learning_action_fn(engine, step, scenario):
        if controller_state.get("scenario") != scenario["scenario"]:
            controller_state.clear()
            controller_state.update(
                {
                    "scenario": scenario["scenario"],
                    "active_actions": {},
                    "green_elapsed_steps": {},
                    "last_green_steps": {
                        intersection_id: {
                            action: 0 for action in spec["actions"]
                        }
                        for intersection_id, spec in intersection_specs.items()
                    },
                    "waiting_durations": {},
                }
            )

        active_actions = controller_state["active_actions"]
        green_elapsed_steps = controller_state["green_elapsed_steps"]
        last_green_steps = controller_state["last_green_steps"]
        waiting_durations = controller_state["waiting_durations"]
        train._update_waiting_durations(engine, waiting_durations, interval)
        lane_waiting_counts = engine.get_lane_waiting_vehicle_count()
        selected_actions = {}
        for intersection_id, spec in intersection_specs.items():
            if (step - 1) % decision_steps == 0:
                red_times = train._red_times_at_step(
                    step,
                    last_green_steps[intersection_id],
                    spec["actions"],
                    interval,
                )
                state = train._queue_state(
                    lane_waiting_counts,
                    spec,
                    red_times,
                    active_actions.get(intersection_id),
                )
                action = train._choose_control_action(
                    q_table,
                    intersection_id,
                    state,
                    spec,
                    step,
                    interval,
                    red_times,
                    active_actions.get(intersection_id),
                    green_elapsed_steps.get(intersection_id, 0),
                    minimum_green_steps,
                    maximum_red_time,
                    0.0,
                    controller_rng,
                )
            else:
                action = active_actions[intersection_id]
            selected_actions[intersection_id] = action
            train._update_phase_timing(
                intersection_id,
                action,
                spec["actions"],
                step,
                active_actions,
                green_elapsed_steps,
                last_green_steps,
            )
        return selected_actions

    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    policy_results = []
    print(
        f"Evaluating all policies on {scenario_count} shared scenarios "
        f"with +/-{demand_variation:.0%} demand variation, "
        f"+/-{vehicle_speed_variation:.0%} vehicle-speed variation, and "
        f"{breakdown_probability:.0%} breakdown probability "
        f"(Q decisions every {decision_steps * interval:g}s, "
        f"maximum red {maximum_red_time:g}s)..."
    )
    for policy_name, action_fn in (
        ("no_training", None),
        ("pretrained_random_forest", rf_action_fn),
        ("q_learning", q_learning_action_fn),
    ):
        metrics, actions = _run_policy(
            config,
            roadnet_path,
            flows,
            scenarios,
            policy_name,
            intersections,
            max_steps,
            thread_num,
            action_fn,
        )
        policy_results.append((metrics, actions))

    all_metrics = pd.concat(
        [result[0] for result in policy_results], ignore_index=True
    )
    all_metrics.to_csv(output_file_path, index=False)
    matrix = _build_action_confusion_matrix(
        policy_results[1][1],
        policy_results[2][1],
        intersections,
        known_actions,
    )
    matrix.to_csv(confusion_matrix_path)

    print("Evaluation summary:")
    for policy_name, (metrics, _) in zip(
        ("no_training", "pretrained_random_forest", "q_learning"),
        policy_results,
    ):
        scenario_results = metrics.groupby("scenario").agg(
            final_average_travel_time=("average_travel_time", "last"),
            peak_queue_length=("total_queue_length", "max"),
        )
        final_average_travel_time = scenario_results[
            "final_average_travel_time"
        ].mean()
        peak_queue_length = scenario_results["peak_queue_length"].mean()
        print(
            f"  {policy_name}: mean final average travel time "
            f"{final_average_travel_time:.2f} seconds across scenarios, "
            f"mean peak queue length {peak_queue_length:.2f}"
        )
    print(
        "Confusion matrix (rows: RF actions, columns: Q-learning actions) "
        f"saved to {confusion_matrix_path}"
    )
    print(f"Step-by-step comparison saved to {output_file_path}")
    return all_metrics, matrix


if __name__ == "__main__":
    run_evaluation()
