from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


@dataclass
class MicroHub:
    name: str
    area: str
    capacity_tph: int
    ideal_vehicle: str
    emission_benefit_pct: float
    notes: str


MICRO_HUBS: List[MicroHub] = [
    MicroHub(
        name="Whitefield EV Consolidation Yard",
        area="Whitefield",
        capacity_tph=120,
        ideal_vehicle="Mini",
        emission_benefit_pct=22.5,
        notes="Works with IT corridor night windows; supports DC fast-charging."
    ),
    MicroHub(
        name="Koramangala Intermediate Hub",
        area="Koramangala",
        capacity_tph=90,
        ideal_vehicle="LCV",
        emission_benefit_pct=18.0,
        notes="Shared dock with tech park; aligns with CBD truck bans."
    ),
    MicroHub(
        name="Hebbal Peripheral Staging",
        area="Hebbal",
        capacity_tph=160,
        ideal_vehicle="MHCV",
        emission_benefit_pct=14.0,
        notes="Link to airport and ORR; ideal for cross-docking before CBD entry."
    ),
]


def microhub_scores(df: pd.DataFrame, selected_area: str) -> List[Dict[str, float]]:
    """Rank micro-hubs by congestion relief and emission impact."""
    congestion = (
        df[df["Area Name"] == selected_area]["Congestion Level"].tail(30).mean()
        if selected_area in df["Area Name"].unique()
        else df["Congestion Level"].tail(30).mean()
    )
    results: List[Dict[str, float]] = []
    for hub in MICRO_HUBS:
        base_score = 50
        if hub.area == selected_area:
            base_score += 15
        relief = max(0.0, min(30.0, (congestion - 60) * 0.3))
        carbon = hub.emission_benefit_pct * 0.4
        total = base_score + relief + carbon
        results.append(
            {
                "name": hub.name,
                "score": round(total, 1),
                "emission_benefit_pct": hub.emission_benefit_pct,
                "notes": hub.notes,
                "ideal_vehicle": hub.ideal_vehicle,
            }
        )
    return sorted(results, key=lambda item: item["score"], reverse=True)
