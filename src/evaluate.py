import json
import os
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


def _run_policy(config, policy_name, intersections, max_steps, thread_num, action_fn):
    policy_config = dict(config)
    policy_config["rlTrafficLight"] = action_fn is not None
    rows = []
    actions_by_step = []

    with tempfile.TemporaryDirectory() as temp_dir:
        engine_config_path = Path(temp_dir) / "config.json"
        with engine_config_path.open("w", encoding="utf-8") as config_file:
            json.dump(policy_config, config_file)

        engine = cityflow.Engine(str(engine_config_path), thread_num=thread_num)
        for step in range(1, max_steps + 1):
            step_actions = {}
            if action_fn is not None:
                action_observations = action_fn(engine)
                for intersection_id in intersections:
                    action = action_observations[intersection_id]
                    engine.set_tl_phase(intersection_id, action)
                    step_actions[intersection_id] = action

            engine.next_step()
            lane_waiting_counts = engine.get_lane_waiting_vehicle_count()
            rows.append(
                {
                    "policy": policy_name,
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
):
    if max_steps <= 0:
        raise ValueError("max_steps must be a positive integer.")

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
    intersection_specs = train._build_intersection_specs(roadnet)
    intersections = list(intersection_specs)
    q_table = train._load_q_table(q_table_file_path, intersection_specs)

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

    def rf_action_fn(engine):
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

    def q_learning_action_fn(engine):
        lane_waiting_counts = engine.get_lane_waiting_vehicle_count()
        return {
            intersection_id: train._greedy_action(
                q_table,
                intersection_id,
                train._queue_state(lane_waiting_counts, spec),
                spec["actions"],
            )
            for intersection_id, spec in intersection_specs.items()
        }

    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    policy_results = []
    for policy_name, action_fn in (
        ("no_training", None),
        ("pretrained_random_forest", rf_action_fn),
        ("q_learning", q_learning_action_fn),
    ):
        metrics, actions = _run_policy(
            config,
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
        final_average_travel_time = metrics["average_travel_time"].iloc[-1]
        peak_queue_length = metrics["total_queue_length"].max()
        print(
            f"  {policy_name}: final average travel time "
            f"{final_average_travel_time:.2f} seconds, "
            f"peak queue length {peak_queue_length}"
        )
    print(
        "Confusion matrix (rows: RF actions, columns: Q-learning actions) "
        f"saved to {confusion_matrix_path}"
    )
    print(f"Step-by-step comparison saved to {output_file_path}")
    return all_metrics, matrix


if __name__ == "__main__":
    run_evaluation()
