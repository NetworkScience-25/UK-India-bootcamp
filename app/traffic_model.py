from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:  # pragma: no cover - import fallback for streamlit script execution
    from .data_utils import get_feature_target_frames
except ImportError:  # pragma: no cover
    from data_utils import get_feature_target_frames


@dataclass
class ModelBundle:
    pipeline: Pipeline
    feature_names: Tuple[str, ...]
    mae: float


NUM_FEATURES = [
    "Traffic Volume",
    "Average Speed",
    "Road Capacity Utilization",
    "Incident Reports",
    "Environmental Impact",
    "Public Transport Usage",
    "Traffic Signal Compliance",
    "Parking Usage",
    "Pedestrian and Cyclist Count",
    "day_of_week",
    "month",
    "is_weekend",
]

CAT_FEATURES = [
    "Area Name",
    "Road/Intersection Name",
    "Weather Conditions",
    "Roadwork and Construction Activity",
]


def train_model(df: pd.DataFrame) -> ModelBundle:
    X, y = get_feature_target_frames(df)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
            (
                "numerical",
                StandardScaler(),
                NUM_FEATURES,
            ),
        ]
    )
    model = RandomForestRegressor(
        n_estimators=220,
        max_depth=12,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_val)
    mae = mean_absolute_error(y_val, predictions)
    bundle = ModelBundle(pipeline=pipeline, feature_names=tuple(X.columns), mae=mae)
    return bundle


def predict_metrics(bundle: ModelBundle, payload: Dict[str, object]) -> Dict[str, float]:
    frame = pd.DataFrame([payload], columns=bundle.feature_names)
    travel_time_index = float(bundle.pipeline.predict(frame)[0])
    congestion_band = np.interp(
        travel_time_index,
        [1.0, 1.2, 1.4, 1.6],
        [30.0, 55.0, 80.0, 100.0],
    )
    eta_modifier_pct = max(0.0, (travel_time_index - 1.0) * 55)
    return {
        "travel_time_index": round(travel_time_index, 3),
        "implied_congestion_pct": round(congestion_band, 1),
        "eta_modifier_pct": round(eta_modifier_pct, 1),
    }
