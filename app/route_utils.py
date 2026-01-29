from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pydeck as pdk
from pyproj import Geod

GEOD = Geod(ellps="WGS84")

AREA_COORDS: Dict[str, Dict[str, float]] = {
    "Indiranagar": {"lat": 12.9784, "lon": 77.6408},
    "Whitefield": {"lat": 12.9698, "lon": 77.7500},
    "Koramangala": {"lat": 12.9352, "lon": 77.6245},
    "M.G. Road": {"lat": 12.9750, "lon": 77.6040},
    "Jayanagar": {"lat": 12.9250, "lon": 77.5938},
    "Hebbal": {"lat": 13.0358, "lon": 77.5970},
    "Yeshwanthpur": {"lat": 13.0170, "lon": 77.5560},
}

_DEFAULT_LAT = 12.9716
_DEFAULT_LON = 77.5946
_MAPBOX_STYLE = "mapbox://styles/mapbox/light-v9"
_FALLBACK_STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"


def _fallback_coords(name: str) -> Dict[str, float]:
    seed = abs(hash(name)) % 10_000
    lat_offset = ((seed % 2000) / 2000.0 - 0.5) * 0.05
    lon_offset = (((seed // 2000) % 2000) / 2000.0 - 0.5) * 0.05
    return {
        "lat": _DEFAULT_LAT + lat_offset,
        "lon": _DEFAULT_LON + lon_offset,
    }


@dataclass
class RouteGeometry:
    start: Dict[str, float]
    end: Dict[str, float]
    distance_km: Optional[float]
    deck: Optional[pdk.Deck]


def get_coords(area: str) -> Dict[str, float]:
    coords = AREA_COORDS.get(area)
    if coords:
        return {"lat": coords["lat"], "lon": coords["lon"]}
    return _fallback_coords(area)


def compute_distance_km(start_area: str, end_area: str) -> Optional[float]:
    start = get_coords(start_area)
    end = get_coords(end_area)
    _, _, dist_m = GEOD.inv(start["lon"], start["lat"], end["lon"], end["lat"])
    distance_km = dist_m / 1000.0
    if distance_km < 0.5:
        return 3.5
    return distance_km


def _resolve_basemap() -> Tuple[str, Optional[str]]:
    mapbox_token = getattr(pdk.settings, "mapbox_api_key", None)
    if mapbox_token:
        return _MAPBOX_STYLE, None
    return _FALLBACK_STYLE, "carto"


def build_route_deck(start_area: str, end_area: str) -> RouteGeometry:
    start = get_coords(start_area)
    end = get_coords(end_area)

    if start == end:
        end = {
            "lat": start["lat"] + 0.015,
            "lon": start["lon"] + 0.015,
        }

    distance_km = compute_distance_km(start_area, end_area)

    line_data: List[Dict[str, List[float]]] = [
        {
            "path": [
                [start["lon"], start["lat"]],
                [end["lon"], end["lat"]],
            ]
        }
    ]
    point_data = [
        {"lon": start["lon"], "lat": start["lat"], "label": "Start"},
        {"lon": end["lon"], "lat": end["lat"], "label": "End"},
    ]

    layers = [
        pdk.Layer(
            "LineLayer",
            line_data,
            get_path="path",
            get_width=6,
            get_color=[24, 128, 232],
            pickable=False,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            point_data,
            get_position="[lon, lat]",
            get_radius=200,
            get_fill_color="[255, 140, 0]",
            pickable=True,
        ),
    ]

    mid_lat = (start["lat"] + end["lat"]) / 2
    mid_lon = (start["lon"] + end["lon"]) / 2

    map_style, map_provider = _resolve_basemap()

    deck_kwargs = {
        "map_style": map_style,
        "initial_view_state": pdk.ViewState(
            latitude=mid_lat,
            longitude=mid_lon,
            zoom=11,
            pitch=40,
        ),
        "layers": layers,
        "tooltip": {"text": "{label}"},
    }
    if map_provider:
        deck_kwargs["map_provider"] = map_provider

    deck = pdk.Deck(**deck_kwargs)

    return RouteGeometry(start=start, end=end, distance_km=distance_km, deck=deck)
