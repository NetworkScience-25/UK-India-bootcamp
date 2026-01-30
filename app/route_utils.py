from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pydeck as pdk
import requests
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
_OSRM_ROUTE_URL = "https://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}"  # noqa: E501


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


def _geodesic_path(start: Dict[str, float], end: Dict[str, float], steps: int = 30) -> List[List[float]]:
    if steps <= 0:
        return [[start["lon"], start["lat"]], [end["lon"], end["lat"]]]
    intermediates = GEOD.npts(start["lon"], start["lat"], end["lon"], end["lat"], steps)
    path = [[start["lon"], start["lat"]]]
    path.extend([[lon, lat] for lon, lat in intermediates])
    path.append([end["lon"], end["lat"]])
    return path


def _fetch_route_path(start: Dict[str, float], end: Dict[str, float]) -> Optional[List[List[float]]]:
    url = _OSRM_ROUTE_URL.format(
        start_lon=start["lon"],
        start_lat=start["lat"],
        end_lon=end["lon"],
        end_lat=end["lat"],
    )
    params = {
        "overview": "full",
        "geometries": "geojson",
    }
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException:
        return None

    routes = payload.get("routes")
    if not routes:
        return None
    first_route = routes[0]
    geometry = first_route.get("geometry", {})
    coordinates = geometry.get("coordinates")
    if not coordinates:
        return None
    return [[float(lon), float(lat)] for lon, lat in coordinates]


def build_route_deck(start_area: str, end_area: str) -> RouteGeometry:
    start = get_coords(start_area)
    end = get_coords(end_area)

    if start == end:
        end = {
            "lat": start["lat"] + 0.015,
            "lon": start["lon"] + 0.015,
        }

    distance_km = compute_distance_km(start_area, end_area)

    route_path = _fetch_route_path(start, end)
    if not route_path:
        route_path = _geodesic_path(start, end)

    line_data: List[Dict[str, List[float]]] = [
        {
            "path": route_path,
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
