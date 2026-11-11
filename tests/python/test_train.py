import json
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

            with patch.object(train.cityflow, "Engine", return_value=engine):
                q_table = train.run_rl_training(
                    config_path=config_path,
                    q_table_path=q_table_path,
                    output_csv=metrics_path,
                    episodes=1,
                    max_steps=2,
                    epsilon_start=1,
                    epsilon_end=1,
                    seed=7,
                )

            self.assertTrue(q_table)
            self.assertTrue(metrics_path.is_file())
            saved_table = joblib.load(q_table_path)
            self.assertEqual(saved_table["format_version"], 1)
            self.assertTrue(saved_table["q_table"])
            self.assertTrue(engine.actions)
            self.assertTrue(all(phase in (0, 2) for _, phase in engine.actions))

    def test_demand_scaling_changes_vehicle_arrival_interval(self):
        scaled = train._scale_flow_demand(
            [{"interval": 5, "route": ["road_a", "road_b"]}],
            demand_scale=1.25,
        )

        self.assertEqual(scaled[0]["interval"], 4)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
