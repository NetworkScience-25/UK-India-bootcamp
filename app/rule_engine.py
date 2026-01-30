from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RouteContext:
    area: str
    road: str
    vehicle_type: str
    planned_hour: int
    day_of_week: int


@dataclass
class Rule:
    name: str
    description: str
    restricted_areas: List[str]
    restricted_roads: List[str]
    restricted_vehicle_types: List[str]
    start_hour: int
    end_hour: int
    days: List[int]
    recommendation: str

    def applies(self, ctx: RouteContext) -> bool:
        area_match = not self.restricted_areas or ctx.area in self.restricted_areas
        road_match = not self.restricted_roads or ctx.road in self.restricted_roads
        vehicle_match = (
            not self.restricted_vehicle_types or ctx.vehicle_type in self.restricted_vehicle_types
        )
        hour_match = self.start_hour <= ctx.planned_hour < self.end_hour
        day_match = not self.days or ctx.day_of_week in self.days
        return area_match and road_match and vehicle_match and hour_match and day_match


CBD_HEAVY_VEHICLE_BAN = Rule(
    name="CBD heavy vehicle curfew",
    description=(
        "Heavy commercial vehicles above 3T banned inside core CBD during peak "
        "windows as per Bengaluru Traffic Police notifications."
    ),
    restricted_areas=[
        "M.G. Road",
        "Indiranagar",
        "Koramangala",
        "Jayanagar",
    ],
    restricted_roads=[],
    restricted_vehicle_types=["LCV", "MHCV", "HCV"],
    start_hour=8,
    end_hour=21,
    days=[0, 1, 2, 3, 4],
    recommendation="Schedule dock transfers pre-08:00 or post-21:00, or hand over to micro-hubs.",
)

NIGHT_ENTRY_ONLY = Rule(
    name="Outer Ring night entry",
    description="Entry for vehicles over 16T limited to 22:00-06:00 beyond ORR toll plazas.",
    restricted_areas=["Hebbal", "Yeshwanthpur"],
    restricted_roads=["Hebbal Flyover", "Tumkur Road"],
    restricted_vehicle_types=["MHCV", "HCV"],
    start_hour=6,
    end_hour=22,
    days=[],
    recommendation="Hold at peripheral yards and use relay fleets during daylight hours.",
)

SCHOOL_ZONE_CAP = Rule(
    name="School zone speed cap",
    description="30 km/h cap applies 07:30-10:00 and 13:30-16:00 on school corridors.",
    restricted_areas=["Indiranagar", "Koramangala"],
    restricted_roads=["CMH Road", "Sarjapur Road"],
    restricted_vehicle_types=["LCV", "MHCV", "HCV"],
    start_hour=7,
    end_hour=16,
    days=[0, 1, 2, 3, 4, 5],
    recommendation="Buffer +12 minutes in ETA or reroute via parallel collectors.",
)

BUS_PRIORITY_LANES = Rule(
    name="Bus priority lane",
    description="Dedicated BMTC priority lanes restrict loading/unloading during commute peaks.",
    restricted_areas=["Whitefield"],
    restricted_roads=["ITPL Main Road"],
    restricted_vehicle_types=["LCV", "MHCV"],
    start_hour=7,
    end_hour=11,
    days=[0, 1, 2, 3, 4],
    recommendation="Switch to electric vans or schedule between 11:00-15:00.",
)

ACTIVE_RULES = [
    CBD_HEAVY_VEHICLE_BAN,
    NIGHT_ENTRY_ONLY,
    SCHOOL_ZONE_CAP,
    BUS_PRIORITY_LANES,
]


def check_route(ctx: RouteContext) -> List[Rule]:
    return [rule for rule in ACTIVE_RULES if rule.applies(ctx)]


def suggest_vehicle_type(payload_tons: float) -> str:
    if payload_tons <= 1.5:
        return "Mini"
    if payload_tons <= 3.5:
        return "LCV"
    if payload_tons <= 7.5:
        return "MHCV"
    return "HCV"


def resolve_vehicle_type(user_choice: Optional[str], payload_tons: float) -> str:
    return user_choice or suggest_vehicle_type(payload_tons)
