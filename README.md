# Bengaluru Commute Companion

A mobile-first logistics planner that blends Bengaluru traffic analytics, regulatory guidance, and micro-hub recommendations into a single Streamlit experience.

## Highlights
- Real-time corridor forecasts derived from the latest records in Banglore_traffic_Dataset.csv
- Random forest regression pipeline that converts historical telemetry into a travel time index and congestion outlook
- Rule engine enforcing local traffic restrictions, with actionable mitigation advice
- Micro-hub recommender that ranks relief yards by congestion impact and carbon savings
- Interactive route visualisation powered by pydeck with synthetic geospatial fallbacks
- In-progress dark-themed mobile shell in app/traffic_app_v2.py that reimagines the planner for touch devices

## Repository Layout
- [app/traffic_app.py](app/traffic_app.py): Primary Streamlit interface featuring trip planning, demand tuning sliders, and compliance gates
- [app/traffic_app_v2.py](app/traffic_app_v2.py): Mobile-first layout prototype with hero cards, CTA stacks, and responsive styling
- [app/traffic_model.py](app/traffic_model.py): RandomForestRegressor training routine with scikit-learn preprocessing pipeline
- [app/data_utils.py](app/data_utils.py): Dataset loader with feature engineering helpers and modelling splits
- [app/rule_engine.py](app/rule_engine.py): Declarative policy rules and context evaluation utilities for Bengaluru corridors
- [app/microhub.py](app/microhub.py): Static micro-hub catalog plus scoring heuristic for congestion and emission relief
- [app/route_utils.py](app/route_utils.py): Pydeck route builder and geodesic distance calculator with fallback coordinate seeding
- [Banglore_traffic_Dataset.csv](Banglore_traffic_Dataset.csv): Historical traffic observations across key corridors
- [requirements.txt](requirements.txt): Python dependencies for the planner and supporting analytics

## Data Primer
[Banglore_traffic_Dataset.csv](Banglore_traffic_Dataset.csv) captures daily metrics for the city's arterial network including volume, speed, incidents, congestion level, compliance, and modal splits. [app/data_utils.py](app/data_utils.py) normalizes dates, derives calendar features, and buckets congestion severity before modeling.

## Model Overview
[app/traffic_model.py](app/traffic_model.py) trains a RandomForestRegressor through a scikit-learn Pipeline:
- ColumnTransformer one-hot encodes categorical corridor descriptors while scaling numerical telemetry
- Train/validation split (80/20) establishes a baseline MAE surfaced in the app startup banner
- predict_metrics() wraps the pipeline to return travel_time_index alongside derived congestion and ETA modifiers for UI consumption

## Operational Logic
Key behaviors inside the Streamlit runtime:
- load_dataframe() caches preprocessed history so subsequent UI interactions remain snappy
- build_feature_payload() assembles up-to-date features for both origin and destination corridors, applying optional incident overrides
- check_route() evaluates all active policy rules; the UI blocks route planning if any restriction triggers
- Route metrics synthesize model predictions, distance estimates from app/route_utils.py, and vehicle-specific drag factors to compute ETA, buffer, and cost breakdowns
- microhub_scores() ranks staging hubs matched to the selected destination for operational handoffs

## Running Locally
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/traffic_app.py
```
The Streamlit server defaults to http://localhost:8501. Use streamlit run app/traffic_app_v2.py to preview the mobile-first experience under construction in [app/traffic_app_v2.py](app/traffic_app_v2.py).

## Using the Planner
1. Pick origin and destination corridors along with payload, vehicle class, and delivery priority
2. Tune departure time and optional incident, volume, or speed assumptions via the expander
3. Review trip forecast metrics, corridor comparisons, and speed trends for situational awareness
4. Inspect regulatory messages inside restriction details whenever the rule engine blocks a trip
5. Explore micro-hub recommendations (v2) to identify viable relay locations in high-congestion windows

## Extending the Project
- Replace [Banglore_traffic_Dataset.csv](Banglore_traffic_Dataset.csv) with refreshed telemetry and re-run the Streamlit app to retrain on launch
- Update [app/rule_engine.py](app/rule_engine.py) with new municipal guidelines or customer-specific service windows
- Expand [app/microhub.py](app/microhub.py) to add hub capacity insight or real-time occupancy feeds
- Wire [app/traffic_app_v2.py](app/traffic_app_v2.py) buttons into backend workflows to trigger dispatch requests or export itineraries
- Integrate live feeds (e.g., HERE, TomTom) by augmenting build_feature_payload() before prediction

## Testing Checklist
- Validate rule coverage by sampling trips across heavy-vehicle windows and ensuring the UI blocks appropriately
- Sanity-check ETA and cost outputs for each vehicle tier against known benchmarks
- Confirm pydeck renders routes for all defined areas and gracefully handles synthetic fallbacks
- Exercise the volume and speed sliders to verify model responses remain within practical bounds
- Smoke-test the mobile shell to ensure CTA visibility and legibility on 375 px viewports

## Roadmap Ideas
- Persist user preferences and favourite corridors for rapid re-planning
- Add fleet-wide dashboards combining multiple corridor predictions
- Incorporate weather and incident live feeds to override historical priors
- Package the model as a FastAPI microservice and consume it from the Streamlit front-ends
- Deploy the Streamlit app on Streamlit Community Cloud or an internal container platform
