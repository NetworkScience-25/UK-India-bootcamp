import math
import pathlib
import sys
from typing import Dict, Optional, Tuple

import pandas as pd
import streamlit as st

APP_DIR = pathlib.Path(__file__).resolve().parent
PARENT_DIR = APP_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))

try:  # pragma: no cover - streamlit path compatibility
    from . import data_utils  # type: ignore
    from .microhub import microhub_scores  # type: ignore
    from .route_utils import build_route_deck  # type: ignore
    from .rule_engine import RouteContext, check_route, resolve_vehicle_type  # type: ignore
    from .traffic_model import predict_metrics, train_model  # type: ignore
except ImportError:  # pragma: no cover
    import data_utils  # type: ignore
    from microhub import microhub_scores  # type: ignore
    from route_utils import build_route_deck  # type: ignore
    from rule_engine import RouteContext, check_route, resolve_vehicle_type  # type: ignore
    from traffic_model import predict_metrics, train_model  # type: ignore

BASE_DIR = PARENT_DIR
DATA_PATH = data_utils.dataset_path(BASE_DIR)

WEEKDAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

VEHICLE_TYPES = ["Mini", "LCV", "MHCV", "HCV"]
VEHICLE_CAPACITY = {
    "Mini": 1.5,
    "LCV": 3.5,
    "MHCV": 7.5,
    "HCV": 12.0,
}
VEHICLE_TIME_FACTOR = {
    "Mini": 1.0,
    "LCV": 1.08,
    "MHCV": 1.18,
    "HCV": 1.28,
}
VEHICLE_PROFILES = {
    "Mini": "City van / pickup suited for quick commerce drops",
    "LCV": "Light commercial vehicle up to 3.5T payload",
    "MHCV": "Multi-axle hauler handling 4-7T loads",
    "HCV": "Heavy cargo carrier best for trunk hauls",
}
VEHICLE_COST_PER_KM = {
    "Mini": 28.0,
    "LCV": 42.0,
    "MHCV": 58.0,
    "HCV": 72.0,
}
VEHICLE_COST_PER_HOUR = {
    "Mini": 320.0,
    "LCV": 450.0,
    "MHCV": 580.0,
    "HCV": 720.0,
}
PRIORITY_WINDOWS = {
    "Standard (4h window)": 240,
    "Express (2h window)": 120,
    "Night linehaul": 480,
}
PRIORITY_SURCHARGE = {
    "Standard (4h window)": 1.0,
    "Express (2h window)": 1.2,
    "Night linehaul": 0.9,
}


@st.cache_data(show_spinner=False)
def load_dataframe() -> pd.DataFrame:
    return data_utils.load_traffic_data(DATA_PATH)


@st.cache_resource(show_spinner=False)
def load_model(df: pd.DataFrame):
    return train_model(df)


def build_feature_payload(
    df: pd.DataFrame,
    feature_names: Tuple[str, ...],
    area: str,
    road: str,
    weekday_index: int,
) -> Tuple[Optional[Dict[str, object]], Optional[pd.Series], pd.DataFrame]:
    corridor = df[(df["Area Name"] == area) & (df["Road/Intersection Name"] == road)].sort_values("Date")
    if corridor.empty:
        return None, None, corridor
    snapshot = corridor.iloc[-1]
    payload: Dict[str, object] = {}
    for col in feature_names:
        if col in snapshot.index:
            payload[col] = snapshot[col]
    payload["Area Name"] = area
    payload["Road/Intersection Name"] = road
    payload["day_of_week"] = weekday_index
    payload["month"] = int(snapshot["Date"].month)
    payload["is_weekend"] = int(weekday_index >= 5)
    return payload, snapshot, corridor.tail(14)


def minutes_to_label(minutes: float) -> str:
    rounded = int(round(minutes))
    hours = rounded // 60
    mins = rounded % 60
    if hours:
        return f"{hours}h {mins}m"
    return f"{mins} min"


def arrival_time_label(depart_hour: int, duration_minutes: float) -> str:
    total_minutes = depart_hour * 60 + int(round(duration_minutes))
    total_minutes %= 24 * 60
    hours = total_minutes // 60
    mins = total_minutes % 60
    return f"{hours:02d}:{mins:02d}"


def signed_minutes_label(minutes: float) -> str:
    if abs(minutes) < 1:
        return "On target"
    prefix = "+" if minutes >= 0 else "-"
    return f"{prefix}{minutes_to_label(abs(minutes))}"


st.set_page_config(page_title="Bengaluru Commute Companion", layout="wide")
st.title("Bengaluru Commute Companion")
st.caption("Personalised traffic outlook for your next intra-city trip.")

with st.spinner("Loading city data"):
    df = load_dataframe()

bundle = load_model(df)
st.success(f"Traffic predictor ready (validation MAE: {bundle.mae:.3f}).")

areas = sorted(df["Area Name"].unique())

if not areas:
    st.error("Dataset is empty. Please verify Banglore_traffic_Dataset.csv is available.")
    st.stop()

with st.sidebar:
    st.header("Plan your ride")
    start_area = st.selectbox("Start area", areas, key="start_area")
    start_roads = sorted(
        df.loc[df["Area Name"] == start_area, "Road/Intersection Name"].unique()
    )
    start_road = st.selectbox("Start road", start_roads, key="start_road")

    destination_options = areas
    default_dest_index = 0
    if len(areas) > 1:
        start_index = areas.index(start_area)
        default_dest_index = (start_index + 1) % len(areas)
    dest_area = st.selectbox(
        "Destination area",
        destination_options,
        index=default_dest_index,
        key="dest_area",
        help="Pick where you are heading. Selecting the same area keeps the trip inside that neighbourhood.",
    )
    dest_roads = sorted(
        df.loc[df["Area Name"] == dest_area, "Road/Intersection Name"].unique()
    )
    dest_road = st.selectbox("Destination road", dest_roads, key="dest_road")

    payload_tons = st.slider("Payload for this trip (tons)", 0.5, 25.0, 3.5, 0.5)
    suggested_vehicle = resolve_vehicle_type(None, payload_tons)
    vehicle_type = st.selectbox(
        "Vehicle class",
        VEHICLE_TYPES,
        index=VEHICLE_TYPES.index(suggested_vehicle),
        help=VEHICLE_PROFILES.get(suggested_vehicle, "Select vehicle matching tonnage"),
    )
    st.caption(f"{vehicle_type}: {VEHICLE_PROFILES.get(vehicle_type, 'Logistics fleet vehicle')}")
    service_priority = st.selectbox(
        "Delivery priority window",
        list(PRIORITY_WINDOWS.keys()),
        index=0,
    )

    planned_hour = st.slider("Preferred departure hour", 0, 23, 9)
    chosen_day = st.selectbox("Travel day", WEEKDAYS, index=0)
    weekday_index = WEEKDAYS.index(chosen_day)

    override_incidents = st.checkbox("Override incident count", value=False)
    incident_reports = (
        st.slider("Incidents along route", 0, 6, 1, key="incident_override")
        if override_incidents
        else None
    )

start_payload, start_snapshot, start_trend = build_feature_payload(
    df, bundle.feature_names, start_area, start_road, weekday_index
)
dest_payload, dest_snapshot, dest_trend = build_feature_payload(
    df, bundle.feature_names, dest_area, dest_road, weekday_index
)

if not start_payload or not dest_payload:
    st.error("Not enough data for the selected start/end roads. Please try another combination.")
    st.stop()

if incident_reports is not None:
    start_payload["Incident Reports"] = incident_reports
    dest_payload["Incident Reports"] = incident_reports

start_volume_value = float(start_snapshot["Traffic Volume"])
dest_volume_value = float(dest_snapshot["Traffic Volume"])
start_speed_value = float(start_snapshot["Average Speed"])
dest_speed_value = float(dest_snapshot["Average Speed"])

with st.expander("Adjust traffic assumptions", expanded=False):
    start_col, dest_col = st.columns(2)
    with start_col:
        start_volume_default = max(1000, min(80000, int(round(start_volume_value / 1000.0)) * 1000))
        start_speed_default = max(5.0, min(80.0, round(start_speed_value, 1)))
        start_volume_value = st.slider(
            "Start corridor volume (veh/day)",
            1000,
            80000,
            start_volume_default,
            1000,
            key="start_volume_slider",
        )
        start_speed_value = st.slider(
            "Start corridor speed (km/h)",
            5.0,
            80.0,
            start_speed_default,
            0.5,
            key="start_speed_slider",
        )
    with dest_col:
        dest_volume_default = max(1000, min(80000, int(round(dest_volume_value / 1000.0)) * 1000))
        dest_speed_default = max(5.0, min(80.0, round(dest_speed_value, 1)))
        dest_volume_value = st.slider(
            "Destination corridor volume (veh/day)",
            1000,
            80000,
            dest_volume_default,
            1000,
            key="dest_volume_slider",
        )
        dest_speed_value = st.slider(
            "Destination corridor speed (km/h)",
            5.0,
            80.0,
            dest_speed_default,
            0.5,
            key="dest_speed_slider",
        )

start_payload["Traffic Volume"] = float(start_volume_value)
start_payload["Average Speed"] = float(start_speed_value)
dest_payload["Traffic Volume"] = float(dest_volume_value)
dest_payload["Average Speed"] = float(dest_speed_value)

start_prediction = predict_metrics(bundle, start_payload)
dest_prediction = predict_metrics(bundle, dest_payload)

vehicle_factor = VEHICLE_TIME_FACTOR.get(vehicle_type, 1.0)

start_ctx = RouteContext(
    area=start_area,
    road=start_road,
    vehicle_type=vehicle_type,
    planned_hour=planned_hour,
    day_of_week=weekday_index,
)
dest_ctx = RouteContext(
    area=dest_area,
    road=dest_road,
    vehicle_type=vehicle_type,
    planned_hour=planned_hour,
    day_of_week=weekday_index,
)
start_restrictions = check_route(start_ctx)
dest_restrictions = check_route(dest_ctx)
active_restrictions = {rule.name: rule for rule in [*start_restrictions, *dest_restrictions]}

if active_restrictions:
    st.error("Trip blocked: regulatory restriction active for the selected slot.")
    with st.expander("Restriction details", expanded=True):
        for rule in active_restrictions.values():
            restricted_types = ", ".join(rule.restricted_vehicle_types) or "All fleet"
            operating_days = (
                ", ".join(WEEKDAYS[idx] for idx in rule.days)
                if rule.days
                else "All days"
            )
            st.write(
                f"- **{rule.name}** ({rule.start_hour:02d}:00–{rule.end_hour:02d}:00 | {operating_days}) \n"
                f"  Restricted vehicles: {restricted_types} \n  Guidance: {rule.recommendation}"
            )
    st.stop()

avg_speed = (float(start_snapshot["Average Speed"]) + float(dest_snapshot["Average Speed"])) / 2
avg_speed = max(avg_speed, 10.0)

route_geom = build_route_deck(start_area, dest_area)
distance_km = route_geom.distance_km or 6.5

route_tti = (start_prediction["travel_time_index"] + dest_prediction["travel_time_index"]) / 2
base_minutes = (distance_km / avg_speed) * 60
adjusted_minutes = base_minutes * route_tti * vehicle_factor
buffer_minutes = max(0.0, adjusted_minutes - base_minutes)

distance_cost = distance_km * VEHICLE_COST_PER_KM.get(vehicle_type, 40.0)
time_cost = (adjusted_minutes / 60.0) * VEHICLE_COST_PER_HOUR.get(vehicle_type, 400.0)
priority_multiplier = PRIORITY_SURCHARGE.get(service_priority, 1.0)
estimated_cost = (distance_cost + time_cost) * priority_multiplier
cost_label = f"INR {estimated_cost:,.0f}"
cost_breakdown = (
    f"Base distance cost: INR {distance_cost:,.0f}\n"
    f"Time cost: INR {time_cost:,.0f}\n"
    f"Priority multiplier: ×{priority_multiplier:.2f}"
)

arrival_label = arrival_time_label(planned_hour, adjusted_minutes)
duration_label = minutes_to_label(adjusted_minutes)
buffer_label = minutes_to_label(buffer_minutes) if buffer_minutes else "No extra buffer"

story_col, map_col = st.columns([1.1, 0.9])

with story_col:
    st.subheader("Trip forecast")
    st.metric("Estimated travel time", duration_label)
    st.metric("Predicted congestion", f"{int(round(route_tti * 60))}% of free-flow")
    st.metric("Suggested buffer", buffer_label)
    st.metric("Estimated cost", cost_label)
    vehicle_drag_pct = int(round((vehicle_factor - 1.0) * 100))
    st.metric("Vehicle drag", f"+{vehicle_drag_pct}%" if vehicle_drag_pct > 0 else "Baseline")

    st.markdown(
        f"Leaving at **{planned_hour:02d}:00** on **{chosen_day}** with an **{vehicle_type.lower()}** "
        f"gets you to {dest_area} around **{arrival_label}**."
    )

    st.caption(cost_breakdown)

    st.markdown("### Corridor outlook")
    corridor_df = pd.DataFrame(
        [
            {
                "Segment": f"{start_area} · {start_road}",
                "Travel time index": start_prediction["travel_time_index"],
                "Congestion %": start_prediction["implied_congestion_pct"],
                "Avg speed (km/h)": round(float(start_snapshot["Average Speed"]), 1),
                "Incidents": int(start_snapshot["Incident Reports"]),
            },
            {
                "Segment": f"{dest_area} · {dest_road}",
                "Travel time index": dest_prediction["travel_time_index"],
                "Congestion %": dest_prediction["implied_congestion_pct"],
                "Avg speed (km/h)": round(float(dest_snapshot["Average Speed"]), 1),
                "Incidents": int(dest_snapshot["Incident Reports"]),
            },
        ]
    )
    st.dataframe(corridor_df, hide_index=True, width="stretch")

    st.markdown("### Recent speed trend")
    combined_trend = (
        pd.concat(
            [
                start_trend.assign(Segment=f"Start · {start_road}"),
                dest_trend.assign(Segment=f"End · {dest_road}"),
            ]
        )
        .set_index("Date")
        [["Average Speed", "Segment"]]
    )
    st.line_chart(combined_trend, y="Average Speed", color="Segment", height=220)

with map_col:
    st.subheader("Route snapshot")
    if route_geom.deck:
        st.pydeck_chart(route_geom.deck, width="stretch")
        st.caption(f"Approximate aerial distance: {distance_km:.1f} km")
    else:
        st.info("Map preview unavailable for the chosen areas.")

st.caption(
    "Tip: Save this page to home screen for quick access before every trip. Predictions refresh on each load."
)
