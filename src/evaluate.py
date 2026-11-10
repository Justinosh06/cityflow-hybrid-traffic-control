import json
import os
import tempfile
from pathlib import Path

import cityflow
import joblib
import pandas as pd


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
    config["rlTrafficLight"] = True
    config["saveReplay"] = False
    return config


def _get_signalized_intersections(config):
    roadnet_path = Path(config["dir"]) / config["roadnetFile"]
    with roadnet_path.open("r", encoding="utf-8") as roadnet_file:
        roadnet = json.load(roadnet_file)

    intersections = [
        intersection["id"]
        for intersection in roadnet.get("intersections", [])
        if not intersection.get("virtual", False)
        and intersection.get("trafficLight")
    ]
    if not intersections:
        raise ValueError(f"No signalized intersections found in {roadnet_path}.")
    return intersections


def run_evaluation(
    config_path="data/config.json",
    model_path="models/rf_pretrain.joblib",
    output_csv="data/evaluation_metrics.csv",
    max_steps=3600,
    thread_num=4,
):
    if max_steps <= 0:
        raise ValueError("max_steps must be a positive integer.")

    config_file_path = _project_path(config_path)
    model_file_path = _project_path(model_path)
    output_file_path = _project_path(output_csv)
    if not config_file_path.is_file():
        raise FileNotFoundError(f"CityFlow config not found: {config_file_path}")
    if not model_file_path.is_file():
        raise FileNotFoundError(f"Trained Random Forest model not found: {model_file_path}")

    config = _load_cityflow_config(config_file_path)
    intersections = _get_signalized_intersections(config)
    model = joblib.load(model_file_path)
    expected_features = getattr(model, "feature_names_in_", None)
    if expected_features is None:
        raise ValueError(
            "The trained model does not expose feature_names_in_; "
            "cannot safely align lane-count features."
        )
    expected_features = list(expected_features)

    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    # CityFlow resolves road and flow files from config["dir"] and can otherwise
    # overwrite replay outputs when saveReplay is enabled in the source config.
    with tempfile.TemporaryDirectory() as temp_dir:
        engine_config_path = Path(temp_dir) / "config.json"
        with engine_config_path.open("w", encoding="utf-8") as config_file:
            json.dump(config, config_file)

        engine = cityflow.Engine(str(engine_config_path), thread_num=thread_num)
        for step in range(1, max_steps + 1):
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

            for intersection_id in intersections:
                engine.set_tl_phase(intersection_id, action)

            engine.next_step()
            lane_waiting_counts = engine.get_lane_waiting_vehicle_count()
            rows.append(
                {
                    "step": step,
                    "total_queue_length": sum(lane_waiting_counts.values()),
                    "average_travel_time": engine.get_average_travel_time(),
                }
            )

    metrics = pd.DataFrame(
        rows, columns=["step", "total_queue_length", "average_travel_time"]
    )
    metrics.to_csv(output_file_path, index=False)

    final_average_travel_time = metrics["average_travel_time"].iloc[-1]
    peak_queue_length = metrics["total_queue_length"].max()
    print(f"Final average travel time: {final_average_travel_time:.2f} seconds")
    print(f"Peak queue length: {peak_queue_length}")
    print(f"Step-by-step metrics saved to {output_file_path}")
    return metrics


if __name__ == "__main__":
    run_evaluation()
