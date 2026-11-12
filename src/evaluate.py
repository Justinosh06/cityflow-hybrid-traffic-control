import json
import math
import os
import random
import shutil
import tempfile
from pathlib import Path

import cityflow
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


def _make_rf_action_fn(
    rf_bundle,
    intersection_specs,
    interval,
    decision_steps,
    minimum_green_steps,
    maximum_red_time,
):
    controller_state = {}

    def action_fn(engine, step, scenario):
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
                }
            )
        active_actions = controller_state["active_actions"]
        green_elapsed_steps = controller_state["green_elapsed_steps"]
        last_green_steps = controller_state["last_green_steps"]
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
                active_action = active_actions.get(intersection_id)
                if train._hold_green_action(
                    active_action,
                    green_elapsed_steps.get(intersection_id, 0),
                    minimum_green_steps,
                    spec["actions"],
                ):
                    action = active_action
                else:
                    overdue_actions = [
                        phase
                        for phase in spec["actions"]
                        if red_times[phase] >= maximum_red_time
                    ]
                    if overdue_actions:
                        action = max(
                            overdue_actions,
                            key=lambda phase: red_times[phase],
                        )
                    else:
                        current_phase = (
                            active_action
                            if active_action is not None
                            else spec["actions"][0]
                        )
                        action = train._rf_greedy_action(
                            rf_bundle,
                            lane_waiting_counts,
                            current_phase,
                            spec["actions"],
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

    return action_fn


def run_evaluation(
    config_path="data/config.json",
    model_path="models/rf_pretrain.joblib",
    q_table_path="models/q_table.joblib",
    cold_q_table_path="models/q_table_cold.joblib",
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
    rf_prior_weight=0.1,
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
    cold_q_table_file_path = _project_path(cold_q_table_path)
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
    if not cold_q_table_file_path.is_file():
        raise FileNotFoundError(
            f"Cold-start Q-learning table not found: "
            f"{cold_q_table_file_path}. Run train.py with --init none first."
        )
    if rf_prior_weight < 0:
        raise ValueError("rf_prior_weight cannot be negative.")

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

    rf_bundle, rf_model_sha256 = train._load_rf_bundle(model_file_path)
    model = rf_bundle["model"]
    known_actions = set(getattr(model, "classes_", []))
    known_actions.update(
        action
        for spec in intersection_specs.values()
        for action in spec["actions"]
    )

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
        init_method="rf",
        rf_prior_weight=rf_prior_weight,
        v0=float(rf_bundle["v0"]),
        rf_model_sha256=rf_model_sha256,
    )
    hybrid_q_table = train._load_q_table(
        q_table_file_path,
        intersection_specs,
        expected_training_settings,
    )
    cold_training_settings = dict(expected_training_settings)
    cold_training_settings["init_method"] = "none"
    cold_q_table = train._load_q_table(
        cold_q_table_file_path,
        intersection_specs,
        cold_training_settings,
    )

    def make_q_action_fn(q_table, init_method):
        controller_state = {}
        controller_rng = random.Random(seed)

        def action_fn(engine, step, scenario):
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
            train._update_waiting_durations(
                engine, waiting_durations, interval
            )
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
                    active_action = active_actions.get(intersection_id)
                    state = train._queue_state(
                        lane_waiting_counts,
                        spec,
                        red_times,
                        active_action,
                    )
                    if init_method == "rf":
                        train._initialize_q_state_from_rf(
                            q_table,
                            intersection_id,
                            state,
                            spec["actions"],
                            rf_bundle,
                            lane_waiting_counts,
                            (
                                active_action
                                if active_action is not None
                                else spec["actions"][0]
                            ),
                            rf_prior_weight,
                        )
                    action = train._choose_control_action(
                        q_table,
                        intersection_id,
                        state,
                        spec,
                        step,
                        interval,
                        red_times,
                        active_action,
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

        return action_fn

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
        ("fixed_time", None),
        (
            "random_forest",
            _make_rf_action_fn(
                rf_bundle,
                intersection_specs,
                interval,
                decision_steps,
                minimum_green_steps,
                maximum_red_time,
            ),
        ),
        ("q_learning_cold", make_q_action_fn(cold_q_table, "none")),
        ("hybrid_rf_q_learning", make_q_action_fn(hybrid_q_table, "rf")),
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
        policy_results.append((policy_name, metrics, actions))

    scenario_summaries = {}
    for result in policy_results:
        policy_name, metrics = result[0], result[1]
        summaries = []
        for scenario_id, scenario_metrics in metrics.groupby("scenario"):
            summaries.append(
                {
                    "policy": policy_name,
                    "scenario": scenario_id,
                    "mean_travel_time": scenario_metrics[
                        "average_travel_time"
                    ].iloc[-1],
                    "mean_peak_queue": scenario_metrics[
                        "total_queue_length"
                    ].max(),
                    "mean_queue_over_hour": scenario_metrics[
                        "total_queue_length"
                    ].mean(),
                }
            )
        scenario_summaries[policy_name] = summaries

    rf_actions = policy_results[1][2]
    hybrid_actions = policy_results[3][2]
    decision_rf_actions = []
    decision_hybrid_actions = []
    agreement_by_scenario = {}
    for scenario_index in range(scenario_count):
        scenario_start = scenario_index * max_steps
        indices = range(
            scenario_start,
            scenario_start + max_steps,
            decision_steps,
        )
        scenario_rf = [rf_actions[index] for index in indices]
        scenario_indices = range(
            scenario_start,
            scenario_start + max_steps,
            decision_steps,
        )
        scenario_hybrid = [hybrid_actions[index] for index in scenario_indices]
        decision_rf_actions.extend(scenario_rf)
        decision_hybrid_actions.extend(scenario_hybrid)
        comparisons = [
            rf_step[intersection_id] == hybrid_step[intersection_id]
            for rf_step, hybrid_step in zip(scenario_rf, scenario_hybrid)
            for intersection_id in intersections
        ]
        agreement_by_scenario[scenario_index + 1] = (
            sum(comparisons) / len(comparisons) if comparisons else float("nan")
        )

    matrix = _build_action_confusion_matrix(
        rf_actions,
        hybrid_actions,
        intersections,
        known_actions,
    )
    summary_rows = []
    for policy_name, summaries in scenario_summaries.items():
        for summary in summaries:
            summary["rf_q_agreement_rate"] = (
                agreement_by_scenario[summary["scenario"]]
                if policy_name == "hybrid_rf_q_learning"
                else float("nan")
            )
            summary_rows.append(summary)
        average = pd.DataFrame(summaries)[
            ["mean_travel_time", "mean_peak_queue", "mean_queue_over_hour"]
        ].mean()
        summary_rows.append(
            {
                "policy": policy_name,
                "scenario": "mean",
                "mean_travel_time": average["mean_travel_time"],
                "mean_peak_queue": average["mean_peak_queue"],
                "mean_queue_over_hour": average["mean_queue_over_hour"],
                "rf_q_agreement_rate": (
                    sum(agreement_by_scenario.values())
                    / len(agreement_by_scenario)
                    if policy_name == "hybrid_rf_q_learning"
                    else float("nan")
                ),
            }
        )
    summary_metrics = pd.DataFrame(summary_rows)
    summary_metrics.to_csv(output_file_path, index=False)
    matrix.to_csv(confusion_matrix_path)

    print("Evaluation summary:")
    for result in policy_results:
        policy_name = result[0]
        average = summary_metrics[
            (summary_metrics["policy"] == policy_name)
            & (summary_metrics["scenario"] == "mean")
        ].iloc[0]
        print(
            f"  {policy_name}: mean travel time "
            f"{average['mean_travel_time']:.2f}s, mean peak queue "
            f"{average['mean_peak_queue']:.2f}, mean hourly queue "
            f"{average['mean_queue_over_hour']:.2f}"
        )
    mean_agreement = sum(agreement_by_scenario.values()) / len(
        agreement_by_scenario
    )
    print(f"RF/Q-table agreement rate: {mean_agreement:.2%}")
    print(
        "Confusion matrix (rows: RF actions, columns: hybrid Q-learning actions) "
        f"saved to {confusion_matrix_path}"
    )
    print(f"Per-scenario policy metrics saved to {output_file_path}")
    return summary_metrics, matrix


if __name__ == "__main__":
    run_evaluation()
