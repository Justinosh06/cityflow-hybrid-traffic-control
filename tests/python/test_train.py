import json
import inspect
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib

from src import evaluate
from src import train


class FakeEngine:
    def __init__(self, *_args, **_kwargs):
        self.step = 0
        self.actions = []
        self.broken_down_vehicles = []
        self.lane_vehicles = {"road_a_0": ["car_1"]}

    def get_lane_waiting_vehicle_count(self):
        return {
            "north_0": max(0, 2 - self.step),
            "south_0": self.step,
        }

    def set_tl_phase(self, intersection_id, phase):
        self.actions.append((intersection_id, phase))

    def next_step(self):
        self.step += 1

    def get_average_travel_time(self):
        return float(self.step)

    def get_lane_vehicles(self):
        return self.lane_vehicles

    def get_vehicle_speed(self):
        return {}

    def set_vehicle_speed(self, vehicle_id, speed):
        self.broken_down_vehicles.append((vehicle_id, speed))


class TestQLearningTraining(unittest.TestCase):
    def test_intersection_uses_long_green_phases_and_incoming_lanes(self):
        specs = train._build_intersection_specs(
            {
                "intersections": [
                    {
                        "id": "junction",
                        "trafficLight": {
                            "lightphases": [
                                {"time": 30, "availableRoadLinks": [0]},
                                {"time": 5, "availableRoadLinks": [0, 1]},
                                {"time": 30, "availableRoadLinks": [1]},
                            ]
                        },
                        "roadLinks": [
                            {
                                "startRoad": "north",
                                "laneLinks": [{"startLaneIndex": 0}],
                            },
                            {
                                "startRoad": "south",
                                "laneLinks": [{"startLaneIndex": 0}],
                            },
                        ],
                    }
                ]
            }
        )

        self.assertEqual(specs["junction"]["actions"], (0, 2))
        self.assertEqual(
            specs["junction"]["incoming_lanes"], ("north_0", "south_0")
        )
        self.assertEqual(
            train._fixed_time_phase(specs["junction"], step=1, interval=1),
            0,
        )
        state = train._queue_state({"north_0": 81}, specs["junction"])
        self.assertEqual(state, (6, 0))

    def test_q_learning_applies_bellman_update(self):
        q_table = {
            "junction": {
                (1,): {0: 4.0},
                (2,): {0: 6.0, 2: 8.0},
            }
        }

        train._update_q_value(
            q_table,
            "junction",
            (1,),
            0,
            reward=3.0,
            next_state=(2,),
            actions=(0, 2),
            learning_rate=0.25,
            discount_factor=0.5,
            terminal=False,
        )

        self.assertEqual(q_table["junction"][(1,)][0], 4.75)

    def test_unseen_states_use_phase_with_longest_queue(self):
        unseen_q_table = {}

        self.assertEqual(
            train._select_action(
                unseen_q_table,
                "junction",
                (1, 5),
                (0, 2),
                epsilon=1,
                rng=random.Random(1),
                fixed_time_action=0,
            ),
            2,
        )
        self.assertEqual(
            train._greedy_action(
                unseen_q_table,
                "junction",
                (1, 5),
                (0, 2),
                fixed_time_action=0,
            ),
            2,
        )

    def test_greedy_action_uses_negative_infinity_for_missing_values(self):
        q_table = {"junction": {(0,): {2: -1.0}}}

        self.assertEqual(
            train._greedy_action(
                q_table,
                "junction",
                (0,),
                (0, 2),
                fixed_time_action=0,
            ),
            2,
        )

    def test_minimum_green_hold(self):
        self.assertTrue(train._hold_green_action(2, 4, 5))
        self.assertFalse(train._hold_green_action(2, 5, 5))
        self.assertFalse(train._hold_green_action(None, 0, 5))

    def test_training_defaults_to_500_episodes(self):
        self.assertEqual(
            inspect.signature(train.run_rl_training).parameters["episodes"].default,
            500,
        )

    def test_requested_training_and_evaluation_defaults(self):
        self.assertEqual(
            inspect.signature(train.run_rl_training)
            .parameters["minimum_green_duration"]
            .default,
            20.0,
        )
        self.assertEqual(
            inspect.signature(evaluate.run_evaluation)
            .parameters["minimum_green_duration"]
            .default,
            20.0,
        )
        self.assertEqual(
            inspect.signature(evaluate.run_evaluation)
            .parameters["breakdown_probability"]
            .default,
            1.0,
        )

    def test_longest_red_phase_is_forced_when_due(self):
        spec = {
            "actions": (0, 2),
            "fixed_time_schedule": ((0, 30), (2, 30)),
        }
        action = train._choose_control_action(
            q_table={"junction": {(0, 0, 0, 0): {0: 100.0, 2: -100.0}}},
            intersection_id="junction",
            state=(0, 0, 0, 0),
            intersection_spec=spec,
            step=81,
            interval=1,
            red_times={0: 20, 2: 80},
            active_action=0,
            green_elapsed_steps=20,
            minimum_green_steps=10,
            maximum_red_time=80,
            epsilon=0,
            rng=random.Random(1),
        )

        self.assertEqual(action, 2)

    def test_reward_penalizes_longest_wait_as_well_as_queue(self):
        reward = train._intersection_reward(
            lane_waiting_counts={"north_0": 2},
            lane_vehicles={"north_0": ["car_1", "car_2"]},
            waiting_durations={"car_1": 40.0, "car_2": 10.0},
            intersection_spec={
                "incoming_lanes": ("north_0",),
            },
            longest_wait_weight=0.1,
        )

        self.assertEqual(reward, -6.0)

    def test_training_updates_and_saves_q_table_without_rf_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            roadnet_path = root / "roadnet.json"
            roadnet_path.write_text(
                json.dumps(
                    {
                        "intersections": [
                            {
                                "id": "junction",
                                "trafficLight": {
                                    "lightphases": [
                                        {"time": 30, "availableRoadLinks": [0]},
                                        {"time": 5, "availableRoadLinks": [0, 1]},
                                        {"time": 30, "availableRoadLinks": [1]},
                                    ]
                                },
                                "roadLinks": [
                                    {
                                        "startRoad": "north",
                                        "laneLinks": [{"startLaneIndex": 0}],
                                    },
                                    {
                                        "startRoad": "south",
                                        "laneLinks": [{"startLaneIndex": 0}],
                                    },
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "dir": str(root),
                        "roadnetFile": roadnet_path.name,
                        "flowFile": "flow.json",
                    }
                ),
                encoding="utf-8",
            )
            (root / "flow.json").write_text("[]", encoding="utf-8")
            q_table_path = root / "q_table.joblib"
            metrics_path = root / "metrics.csv"
            engine = FakeEngine()
            joblib.dump(
                {
                    "format_version": train.Q_TABLE_FORMAT_VERSION,
                    "actions": {"junction": [0, 2]},
                    "q_table": {"junction": {(99, 99): {0: -100.0}}},
                },
                q_table_path,
            )

            with (
                patch.object(train.cityflow, "Engine", return_value=engine),
                patch.object(
                    train,
                    "_update_q_value",
                    wraps=train._update_q_value,
                ) as q_update,
            ):
                q_table = train.run_rl_training(
                    config_path=config_path,
                    q_table_path=q_table_path,
                    output_csv=metrics_path,
                    episodes=1,
                    max_steps=22,
                    minimum_green_duration=2,
                    decision_interval=10,
                    epsilon_start=1,
                    epsilon_end=1,
                    seed=7,
                )

            self.assertTrue(q_table)
            self.assertTrue(metrics_path.is_file())
            self.assertEqual(q_update.call_count, 3)
            saved_table = joblib.load(q_table_path)
            self.assertEqual(saved_table["format_version"], 3)
            self.assertEqual(
                saved_table["training_settings"]["decision_steps"], 10
            )
            self.assertIsNone(
                saved_table["training_settings"]["red_time_thresholds_seconds"]
            )
            self.assertTrue(saved_table["q_table"])
            self.assertNotIn((99, 99), saved_table["q_table"]["junction"])
            self.assertTrue(engine.actions)
            self.assertTrue(all(phase in (0, 2) for _, phase in engine.actions))
            self.assertEqual(engine.actions[:2], [("junction", 0)] * 2)

    def test_demand_scaling_changes_vehicle_arrival_interval(self):
        scaled = train._scale_flow_demand(
            [
                {
                    "interval": 5,
                    "route": ["road_a", "road_b"],
                    "vehicle": {"maxSpeed": 10},
                }
            ],
            demand_scale=1.25,
            vehicle_speed_scale=0.8,
        )

        self.assertEqual(scaled[0]["interval"], 4)
        self.assertEqual(scaled[0]["vehicle"]["maxSpeed"], 8)

    def test_scenario_generator_includes_speed_and_breakdown_uncertainties(self):
        scenario = train._generate_scenario_uncertainties(
            {
                "roads": [
                    {"id": "road_a", "lanes": [{}, {}]},
                ]
            },
            max_steps=10,
            rng=random.Random(3),
            demand_variation=0.2,
            vehicle_speed_variation=0.15,
            breakdown_probability=1,
            breakdown_duration_steps=4,
        )

        self.assertGreater(scenario["demand_scale"], 0)
        self.assertGreater(scenario["vehicle_speed_scale"], 0)
        self.assertEqual(len(scenario["incidents"]), 1)
        self.assertIn(
            scenario["incidents"][0]["lane_id"], {"road_a_0", "road_a_1"}
        )
        self.assertEqual(scenario["incidents"][0]["type"], "vehicle_breakdown")

    def test_breakdown_does_not_apply_speed_to_vehicle_that_left_lane(self):
        engine = FakeEngine()
        incident = [
            {
                "type": "vehicle_breakdown",
                "lane_id": "road_a_0",
                "start_step": 1,
                "duration_steps": 3,
            }
        ]
        affected_vehicles = {}

        train._apply_scenario_incidents(engine, incident, 1, affected_vehicles)
        engine.lane_vehicles = {"road_a_0": []}
        train._apply_scenario_incidents(engine, incident, 2, affected_vehicles)
        self.assertNotIn("road_a_0", affected_vehicles)

        engine.lane_vehicles = {"road_a_0": ["car_2"]}
        train._apply_scenario_incidents(engine, incident, 3, affected_vehicles)

        self.assertEqual(
            engine.broken_down_vehicles, [("car_1", 0), ("car_2", 0)]
        )

    def test_confusion_matrix_compares_q_actions_to_rf_actions(self):
        rf_actions = [{"junction": 0}, {"junction": 2}, {"junction": 0}]
        q_actions = [{"junction": 2}, {"junction": 2}, {"junction": 0}]

        matrix = evaluate._build_action_confusion_matrix(
            rf_actions,
            q_actions,
            ["junction"],
            known_actions={0, 1, 2, 3},
        )

        self.assertEqual(list(matrix.index), [0, 1, 2, 3])
        self.assertEqual(matrix.loc[0, 0], 1)
        self.assertEqual(matrix.loc[0, 2], 1)
        self.assertEqual(matrix.loc[2, 2], 1)
        self.assertEqual(int(matrix.to_numpy().sum()), 3)

    def test_evaluation_policies_reuse_identical_demand_and_seeds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            roadnet_path = root / "roadnet.json"
            roadnet_path.write_text(
                json.dumps(
                    {
                        "roads": [
                            {"id": "road_a", "lanes": [{}]},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            flows = [
                {
                    "interval": 5,
                    "route": ["road_a", "road_b"],
                    "vehicle": {"maxSpeed": 10},
                }
            ]
            config = {"dir": str(root), "saveReplay": False}
            scenarios = evaluate._build_evaluation_scenarios(
                json.loads(roadnet_path.read_text(encoding="utf-8")),
                scenario_count=2,
                max_steps=2,
                demand_variation=0.2,
                vehicle_speed_variation=0.15,
                breakdown_probability=1,
                breakdown_duration_steps=2,
                seed=3,
            )
            engine_inputs = []

            def create_engine(config_path, thread_num):
                with Path(config_path).open(encoding="utf-8") as file:
                    engine_config = json.load(file)
                with (
                    Path(engine_config["dir"]) / engine_config["flowFile"]
                ).open(encoding="utf-8") as file:
                    episode_flows = json.load(file)
                engine = FakeEngine()
                engine_inputs.append(
                    (
                        engine_config["seed"],
                        episode_flows,
                        engine_config["rlTrafficLight"],
                        engine,
                    )
                )
                return engine

            action_fn = lambda _engine, _step, _scenario: {"junction": 0}
            with patch.object(
                evaluate.cityflow, "Engine", side_effect=create_engine
            ):
                for policy_name, policy_action in (
                    ("no_training", None),
                    ("pretrained", action_fn),
                ):
                    evaluate._run_policy(
                        config,
                        roadnet_path,
                        flows,
                        scenarios,
                        policy_name,
                        ["junction"],
                        max_steps=2,
                        thread_num=1,
                        action_fn=policy_action,
                    )

            baseline_inputs = engine_inputs[:2]
            trained_inputs = engine_inputs[2:]
            self.assertEqual(
                [item[:2] for item in baseline_inputs],
                [item[:2] for item in trained_inputs],
            )
            self.assertTrue(all(not item[2] for item in baseline_inputs))
            self.assertTrue(all(item[2] for item in trained_inputs))
            self.assertTrue(
                all(item["incidents"] for item in scenarios)
            )
            self.assertTrue(
                all(item[3].broken_down_vehicles for item in engine_inputs)
            )
            self.assertEqual(
                [
                    (item[0], item[1])
                    for item in baseline_inputs
                ],
                [
                    (item[0], item[1])
                    for item in trained_inputs
                ],
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
