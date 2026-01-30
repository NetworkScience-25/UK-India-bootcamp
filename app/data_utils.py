import pathlib
from typing import Tuple

import pandas as pd

DATASET_FILENAME = "Banglore_traffic_Dataset.csv"


def dataset_path(base_dir: pathlib.Path) -> pathlib.Path:
    """Return absolute path to the traffic dataset."""
    return (base_dir / DATASET_FILENAME).resolve()


def load_traffic_data(csv_path: pathlib.Path) -> pd.DataFrame:
    """Read and lightly preprocess the traffic dataset."""
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["congestion_bucket"] = pd.cut(
        df["Congestion Level"],
        bins=[-1, 50, 80, 100, 101],
        labels=["Low", "Moderate", "High", "Severe"],
    )
    return df


def get_feature_target_frames(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split model features and regression target."""
    feature_cols = [
        "Area Name",
        "Road/Intersection Name",
        "Traffic Volume",
        "Average Speed",
        "Road Capacity Utilization",
        "Incident Reports",
        "Environmental Impact",
        "Public Transport Usage",
        "Traffic Signal Compliance",
        "Parking Usage",
        "Pedestrian and Cyclist Count",
        "Weather Conditions",
        "Roadwork and Construction Activity",
        "day_of_week",
        "month",
        "is_weekend",
    ]
    X = df[feature_cols].copy()
    y = df["Travel Time Index"].astype(float)
    return X, y
