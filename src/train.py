import json
import os
import random
import shutil
import tempfile
from bisect import bisect_right
from pathlib import Path

import cityflow
import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUEUE_THRESHOLDS = (0, 2, 5, 10, 20)
Q_TABLE_FORMAT_VERSION = 1


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


def _build_intersection_specs(roadnet):
    specs = {}
    for intersection in roadnet.get("intersections", []):
        traffic_light = intersection.get("trafficLight")
        if intersection.get("virtual", False) or not traffic_light:
            continue

        road_links = intersection.get("roadLinks", [])
        phases = traffic_light.get("lightphases", [])
        phase_lanes = {}
        phase_durations = {}
        for phase_index, phase in enumerate(phases):
            lane_ids = set()
            for road_link_index in phase.get("availableRoadLinks", []):
                if road_link_index < 0 or road_link_index >= len(road_links):
                    raise ValueError(
                        f"Invalid road link index {road_link_index} in "
                        f"intersection {intersection['id']}."
                    )
                road_link = road_links[road_link_index]
                for lane_link in road_link.get("laneLinks", []):
                    lane_ids.add(
                        f"{road_link['startRoad']}_{lane_link['startLaneIndex']}"
                    )
            if lane_ids:
                phase_lanes[phase_index] = tuple(sorted(lane_ids))
                phase_durations[phase_index] = phase.get("time", 0)

        if not phase_lanes:
            raise ValueError(
                f"No controllable traffic-light phases found for "
                f"intersection {intersection['id']}."
            )

        longest_phase = max(phase_durations.values())
        actions = tuple(
            phase_index
            for phase_index, duration in phase_durations.items()
            if duration == longest_phase
        )
        incoming_lanes = tuple(
            sorted(
                {
                    lane_id
                    for phase_index in actions
                    for lane_id in phase_lanes[phase_index]
                }
            )
        )
        specs[intersection["id"]] = {
            "actions": actions,
            "phase_lanes": {
                phase_index: phase_lanes[phase_index] for phase_index in actions
            },
            "incoming_lanes": incoming_lanes,
        }

    if not specs:
        raise ValueError("No signalized intersections found in the roadnet.")
    return specs


def _queue_state(lane_waiting_counts, intersection_spec):
    return tuple(
        bisect_right(
            QUEUE_THRESHOLDS,
            sum(lane_waiting_counts.get(lane_id, 0) for lane_id in lane_ids),
        )
        for lane_ids in intersection_spec["phase_lanes"].values()
    )


def _intersection_queue(lane_waiting_counts, intersection_spec):
    return sum(
        lane_waiting_counts.get(lane_id, 0)
        for lane_id in intersection_spec["incoming_lanes"]
    )


def _select_action(q_table, intersection_id, state, actions, epsilon, rng):
    if rng.random() < epsilon:
        return rng.choice(actions)

    state_values = (
        q_table.get(intersection_id, {})
        .get(state, {})
    )
    best_value = max(state_values.get(action, 0.0) for action in actions)
    best_actions = [
        action
        for action in actions
        if state_values.get(action, 0.0) == best_value
    ]
    return rng.choice(best_actions)


def _greedy_action(q_table, intersection_id, state, actions):
    state_values = q_table.get(intersection_id, {}).get(state, {})
    return max(actions, key=lambda action: state_values.get(action, 0.0))


def _update_q_value(
    q_table,
    intersection_id,
    state,
    action,
    reward,
    next_state,
    actions,
    learning_rate,
    discount_factor,
    terminal,
):
    intersection_table = q_table.setdefault(intersection_id, {})
    state_values = intersection_table.setdefault(state, {})
    old_value = state_values.get(action, 0.0)
    next_values = intersection_table.get(next_state, {})
    future_value = (
        0.0
        if terminal
        else max(next_values.get(next_action, 0.0) for next_action in actions)
    )
    state_values[action] = old_value + learning_rate * (
        reward + discount_factor * future_value - old_value
    )


def _load_q_table(q_table_path, intersection_specs):
    if not q_table_path.is_file():
        return {}

    payload = joblib.load(q_table_path)
    if (
        not isinstance(payload, dict)
        or payload.get("format_version") != Q_TABLE_FORMAT_VERSION
        or not isinstance(payload.get("q_table"), dict)
    ):
        raise ValueError(f"Unsupported Q-table format in {q_table_path}.")

    saved_actions = payload.get("actions")
    current_actions = {
        intersection_id: list(spec["actions"])
        for intersection_id, spec in intersection_specs.items()
    }
    if saved_actions != current_actions:
        raise ValueError(
            "The saved Q-table's intersection phases do not match the current "
            "roadnet. Choose a new q_table_path to start fresh."
        )
    return payload["q_table"]


def _scale_flow_demand(flows, demand_scale):
    scaled_flows = json.loads(json.dumps(flows))
    for flow in scaled_flows:
        interval = flow.get("interval")
        if not isinstance(interval, (int, float)) or interval <= 0:
            raise ValueError("Every traffic flow must have a positive interval.")
        flow["interval"] = interval / demand_scale
    return scaled_flows


def run_rl_training(
    config_path="data/config.json",
    q_table_path="models/q_table.joblib",
    episodes=5,
    max_steps=3600,
    output_csv="data/training_metrics.csv",
    thread_num=4,
    learning_rate=0.1,
    discount_factor=0.95,
    epsilon_start=1.0,
    epsilon_end=0.05,
    demand_variation=0.2,
    seed=0,
):
    if episodes <= 0:
        raise ValueError("episodes must be a positive integer.")
    if max_steps <= 0:
        raise ValueError("max_steps must be a positive integer.")
    if not 0 < learning_rate <= 1:
        raise ValueError("learning_rate must be in the range (0, 1].")
    if not 0 <= discount_factor <= 1:
        raise ValueError("discount_factor must be in the range [0, 1].")
    if not 0 <= epsilon_end <= epsilon_start <= 1:
        raise ValueError(
            "epsilon values must satisfy 0 <= epsilon_end <= epsilon_start <= 1."
        )
    if not 0 <= demand_variation < 1:
        raise ValueError("demand_variation must be in the range [0, 1).")

    config_file_path = _project_path(config_path)
    q_table_file_path = _project_path(q_table_path)
    output_file_path = _project_path(output_csv)
    if not config_file_path.is_file():
        raise FileNotFoundError(f"CityFlow config not found: {config_file_path}")

    config = _load_cityflow_config(config_file_path)
    roadnet_path = Path(config["dir"]) / config["roadnetFile"]
    with roadnet_path.open("r", encoding="utf-8") as roadnet_file:
        roadnet = json.load(roadnet_file)
    flow_path = Path(config["dir"]) / config["flowFile"]
    with flow_path.open("r", encoding="utf-8") as flow_file:
        flows = json.load(flow_file)
    intersection_specs = _build_intersection_specs(roadnet)
    q_table = _load_q_table(q_table_file_path, intersection_specs)

    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    q_table_file_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    print(
        f"Starting Q-Learning for {episodes} episodes "
        f"with +/-{demand_variation:.0%} traffic-demand variation..."
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        total_training_steps = episodes * max_steps
        for episode in range(1, episodes + 1):
            demand_scale = rng.uniform(
                1 - demand_variation, 1 + demand_variation
            )
            episode_dir = Path(temp_dir) / f"episode_{episode}"
            episode_dir.mkdir()
            shutil.copy2(roadnet_path, episode_dir / "roadnet.json")
            with (episode_dir / "flow.json").open(
                "w", encoding="utf-8"
            ) as flow_file:
                json.dump(_scale_flow_demand(flows, demand_scale), flow_file)

            episode_config = dict(config)
            episode_config.update(
                {
                    "dir": str(episode_dir.resolve()) + os.sep,
                    "roadnetFile": "roadnet.json",
                    "flowFile": "flow.json",
                    "seed": rng.randrange(0, 2**31),
                }
            )
            engine_config_path = episode_dir / "config.json"
            with engine_config_path.open("w", encoding="utf-8") as config_file:
                json.dump(episode_config, config_file)
            engine = cityflow.Engine(
                str(engine_config_path), thread_num=thread_num
            )

            episode_rows = []
            lane_waiting_counts = engine.get_lane_waiting_vehicle_count()
            for step in range(1, max_steps + 1):
                current_states = {
                    intersection_id: _queue_state(
                        lane_waiting_counts, intersection_spec
                    )
                    for intersection_id, intersection_spec in intersection_specs.items()
                }
                elapsed_steps = (episode - 1) * max_steps + step - 1
                progress = elapsed_steps / max(total_training_steps - 1, 1)
                epsilon = epsilon_start + progress * (epsilon_end - epsilon_start)

                selected_actions = {}
                for intersection_id, intersection_spec in intersection_specs.items():
                    action = _select_action(
                        q_table,
                        intersection_id,
                        current_states[intersection_id],
                        intersection_spec["actions"],
                        epsilon,
                        rng,
                    )
                    selected_actions[intersection_id] = action
                    engine.set_tl_phase(intersection_id, action)

                engine.next_step()
                next_lane_waiting_counts = engine.get_lane_waiting_vehicle_count()
                terminal = step == max_steps
                for intersection_id, intersection_spec in intersection_specs.items():
                    next_state = _queue_state(
                        next_lane_waiting_counts, intersection_spec
                    )
                    reward = -_intersection_queue(
                        next_lane_waiting_counts, intersection_spec
                    )
                    _update_q_value(
                        q_table,
                        intersection_id,
                        current_states[intersection_id],
                        selected_actions[intersection_id],
                        reward,
                        next_state,
                        intersection_spec["actions"],
                        learning_rate,
                        discount_factor,
                        terminal,
                    )

                lane_waiting_counts = next_lane_waiting_counts
                episode_rows.append(
                    {
                        "episode": episode,
                        "step": step,
                        "demand_scale": demand_scale,
                        "total_queue_length": sum(next_lane_waiting_counts.values()),
                        "average_travel_time": engine.get_average_travel_time(),
                    }
                )

            episode_metrics = pd.DataFrame(episode_rows)
            episode_metrics.to_csv(
                output_file_path,
                index=False,
                mode="w" if episode == 1 else "a",
                header=episode == 1,
            )
            final_average_travel_time = episode_metrics[
                "average_travel_time"
            ].iloc[-1]
            peak_queue_length = episode_metrics["total_queue_length"].max()
            print(
                f"Episode {episode}/{episodes} - "
                f"final average travel time: {final_average_travel_time:.2f} seconds, "
                f"peak queue length: {peak_queue_length}"
            )

    joblib.dump(
        {
            "format_version": Q_TABLE_FORMAT_VERSION,
            "actions": {
                intersection_id: list(spec["actions"])
                for intersection_id, spec in intersection_specs.items()
            },
            "q_table": q_table,
        },
        q_table_file_path,
    )
    print(f"Learned Q-table saved to {q_table_file_path}")
    print(f"Step-by-step training metrics saved to {output_file_path}")
    return q_table


if __name__ == "__main__":
    run_rl_training()
