import json
import math
import os
import random
import shutil
import tempfile
from pathlib import Path

import cityflow
import pandas as pd

try:
    from . import train
except ImportError:
    import train


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _project_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _teacher_action(
    spec,
    step,
    interval,
    active_action,
    green_elapsed_steps,
    last_green_steps,
    minimum_green_steps,
    maximum_red_time,
    lane_waiting_counts,
):
    if train._hold_green_action(
        active_action,
        green_elapsed_steps,
        minimum_green_steps,
        spec["actions"],
    ):
        return active_action

    red_times = train._red_times_at_step(
        step, last_green_steps, spec["actions"], interval
    )
    overdue_actions = [
        action
        for action in spec["actions"]
        if red_times[action] >= maximum_red_time
    ]
    if overdue_actions:
        return max(overdue_actions, key=lambda action: red_times[action])

    return min(
        spec["actions"],
        key=lambda action: (
            -sum(
                lane_waiting_counts.get(lane_id, 0)
                for lane_id in spec["phase_lanes"][action]
            ),
            action,
        ),
    )


def run_lqf_collection(
    config_path="data/config.json",
    output_csv="data/lqf_dataset.csv",
    max_steps=3600,
    episodes=30,
    thread_num=4,
    minimum_green_duration=20.0,
    decision_interval=10.0,
    maximum_red_time=80.0,
    longest_wait_weight=0.1,
    demand_variation=0.2,
    vehicle_speed_variation=0.15,
    breakdown_probability=0.5,
    breakdown_duration_steps=60,
    seed=2026,
):
    config_file_path = _project_path(config_path)
    output_file_path = _project_path(output_csv)
    if not config_file_path.is_file():
        raise FileNotFoundError(f"CityFlow config not found: {config_file_path}")
    if episodes <= 0 or max_steps <= 0:
        raise ValueError("episodes and max_steps must be positive.")
    if decision_interval <= 0 or minimum_green_duration <= 0:
        raise ValueError("Decision interval and minimum green must be positive.")
    if maximum_red_time <= 0:
        raise ValueError("maximum_red_time must be positive.")

    config = train._load_cityflow_config(config_file_path)
    roadnet_path = Path(config["dir"]) / config["roadnetFile"]
    flow_path = Path(config["dir"]) / config["flowFile"]
    roadnet = json.loads(roadnet_path.read_text(encoding="utf-8"))
    flows = json.loads(flow_path.read_text(encoding="utf-8"))
    intersection_specs = train._build_intersection_specs(roadnet)
    intersection_id = "intersection_1_1"
    if intersection_id not in intersection_specs:
        raise ValueError(
            f"Expected signalized intersection {intersection_id!r} in roadnet."
        )
    spec = intersection_specs[intersection_id]
    interval = config.get("interval", 1.0)
    if interval <= 0:
        raise ValueError("CityFlow interval must be positive.")
    decision_steps = max(1, math.ceil(decision_interval / interval))
    minimum_green_steps = max(
        1, math.ceil(minimum_green_duration / interval)
    )

    lane_names = sorted(
        f"{road['id']}_{lane_index}"
        for road in roadnet.get("roads", [])
        for lane_index in range(len(road.get("lanes", [])))
    )
    if not lane_names:
        raise ValueError("No lanes found in the roadnet.")

    rng = random.Random(seed)
    rows = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for episode in range(1, episodes + 1):
            scenario = train._generate_scenario_uncertainties(
                roadnet,
                max_steps,
                rng,
                demand_variation,
                vehicle_speed_variation,
                breakdown_probability,
                breakdown_duration_steps,
            )
            episode_dir = Path(temp_dir) / f"episode_{episode}"
            episode_dir.mkdir()
            shutil.copy2(roadnet_path, episode_dir / "roadnet.json")
            (episode_dir / "flow.json").write_text(
                json.dumps(
                    train._scale_flow_demand(
                        flows,
                        scenario["demand_scale"],
                        scenario["vehicle_speed_scale"],
                    )
                ),
                encoding="utf-8",
            )
            episode_config = dict(config)
            episode_config.update(
                {
                    "dir": str(episode_dir.resolve()) + os.sep,
                    "roadnetFile": "roadnet.json",
                    "flowFile": "flow.json",
                    "seed": scenario["seed"],
                    "rlTrafficLight": True,
                }
            )
            engine_config_path = episode_dir / "config.json"
            engine_config_path.write_text(
                json.dumps(episode_config), encoding="utf-8"
            )
            engine = cityflow.Engine(
                str(engine_config_path), thread_num=thread_num
            )

            active_actions = {}
            green_elapsed_steps = {}
            last_green_steps = {action: 0 for action in spec["actions"]}
            waiting_durations = {}
            affected_vehicles = {}
            lane_waiting_counts = engine.get_lane_waiting_vehicle_count()
            decision_row = None
            decision_reward = 0.0
            for step in range(1, max_steps + 1):
                train._apply_scenario_incidents(
                    engine, scenario["incidents"], step, affected_vehicles
                )
                is_decision_step = (step - 1) % decision_steps == 0
                if is_decision_step:
                    current_phase = active_actions.get(
                        intersection_id, spec["actions"][0]
                    )
                    action = _teacher_action(
                        spec,
                        step,
                        interval,
                        active_actions.get(intersection_id),
                        green_elapsed_steps.get(intersection_id, 0),
                        last_green_steps,
                        minimum_green_steps,
                        maximum_red_time,
                        lane_waiting_counts,
                    )
                    decision_row = {
                        "episode": episode,
                        "step": step - 1,
                        **{
                            lane_id: lane_waiting_counts.get(lane_id, 0)
                            for lane_id in lane_names
                        },
                        "current_phase": current_phase,
                        "action": action,
                    }
                    decision_reward = 0.0
                    train._update_phase_timing(
                        intersection_id,
                        action,
                        spec["actions"],
                        step,
                        active_actions,
                        green_elapsed_steps,
                        {intersection_id: last_green_steps},
                    )

                engine.set_tl_phase(
                    intersection_id, active_actions[intersection_id]
                )
                engine.next_step()
                next_lane_waiting_counts = (
                    engine.get_lane_waiting_vehicle_count()
                )
                train._update_waiting_durations(
                    engine, waiting_durations, interval
                )
                lane_vehicles = engine.get_lane_vehicles()
                decision_reward += train._intersection_reward(
                    next_lane_waiting_counts,
                    lane_vehicles,
                    waiting_durations,
                    spec,
                    longest_wait_weight,
                )
                if not is_decision_step:
                    action = active_actions[intersection_id]
                    if action in spec["actions"]:
                        last_green_steps[action] = step
                    green_elapsed_steps[intersection_id] += 1

                interval_complete = (
                    step % decision_steps == 0 or step == max_steps
                )
                if interval_complete:
                    decision_row["reward"] = decision_reward
                    rows.append(decision_row)
                lane_waiting_counts = next_lane_waiting_counts

            print(f"Collected expert episode {episode}/{episodes}.")

    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_file_path, index=False)
    lane_list_path = output_file_path.with_name(
        f"{output_file_path.stem}_lanes.json"
    )
    lane_list_path.write_text(
        json.dumps(lane_names, indent=2), encoding="utf-8"
    )
    print(
        f"Saved {len(rows)} LQF decisions to {output_file_path}; "
        f"lane names saved to {lane_list_path}."
    )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    run_lqf_collection()
