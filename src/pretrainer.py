from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit


def train_random_forest(
    dataset_path="data/lqf_dataset.csv",
    model_output_path="models/rf_pretrain.joblib",
    discount_factor=0.95,
):
    dataset_file_path = Path(dataset_path)
    if not dataset_file_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_file_path}. Run collector.py first!"
        )
    if discount_factor != 0.95:
        raise ValueError("The RF pretraining return uses discount 0.95.")

    df = pd.read_csv(dataset_file_path)
    required_columns = {"episode", "step", "current_phase", "action", "reward"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )
    episode_count = int(df["episode"].nunique())
    if episode_count != 30:
        raise ValueError(
            f"Expected 30 teacher episodes, found {episode_count}."
        )
    expected_actions = {0, 2, 4, 6}
    if set(df["action"].unique()) != expected_actions:
        raise ValueError(
            "The expert dataset must contain all and only actions 0, 2, 4, 6."
        )
    lane_names = [
        column
        for column in df.columns
        if column not in required_columns
    ]
    feature_names = [*lane_names, "current_phase"]
    if not lane_names:
        raise ValueError("Dataset must contain at least one lane count feature.")

    features = df[feature_names]
    labels = df["action"]
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=42
    )
    train_indices, test_indices = next(
        splitter.split(features, labels, groups=df["episode"])
    )
    train_episodes = set(df.iloc[train_indices]["episode"])
    test_episodes = set(df.iloc[test_indices]["episode"])
    if train_episodes & test_episodes:
        raise RuntimeError("Episode groups leaked between train and test splits.")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(features.iloc[train_indices], labels.iloc[train_indices])
    predictions = model.predict(features.iloc[test_indices])
    accuracy = accuracy_score(labels.iloc[test_indices], predictions)
    print(f"Test accuracy: {accuracy:.4f}")
    print(
        "Per-class report:\n"
        + classification_report(
            labels.iloc[test_indices],
            predictions,
            labels=[0, 2, 4, 6],
            zero_division=0,
        )
    )

    importances = pd.Series(
        model.feature_importances_, index=feature_names
    ).sort_values(ascending=False)
    print("Top 10 feature importances:")
    for feature_name, importance in importances.head(10).items():
        print(f"  {feature_name}: {importance:.6f}")

    returns = np.empty(len(df), dtype=float)
    for indices in df.groupby("episode", sort=False).groups.values():
        ordered_indices = sorted(indices, key=lambda index: df.at[index, "step"])
        discounted_return = 0.0
        for index in reversed(ordered_indices):
            discounted_return = (
                float(df.at[index, "reward"])
                + discount_factor * discounted_return
            )
            returns[index] = discounted_return
    v0 = float(returns.mean())
    print(f"V0 (mean discounted episode return): {v0:.6f}")

    output_path = Path(model_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "v0": v0,
            "gamma": discount_factor,
            "teacher": "LQF",
            "episodes": episode_count,
        },
        output_path,
    )
    print(f"Model bundle saved to {output_path}")
    return model


if __name__ == "__main__":
    train_random_forest()
