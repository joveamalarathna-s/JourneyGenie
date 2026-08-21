"""
JourneyGenie — AI Personalized Tourist Guide
Team: SHE CODES | Smart India Hackathon 2026 | Problem Statement TC-S01

A single-file Streamlit MVP that uses the Google Gemini API to generate
a personalized, budget-aware, multi-day travel itinerary, renders it as
timeline cards with safety advisories and traffic estimates, plots it on
an interactive Folium map, and shows a budget usage breakdown.

NOTE on "live" data:
- Real-time traffic requires a paid Google Maps Traffic API with billing
  enabled, which is outside the scope of a free hackathon MVP. Instead,
  each stop gets an AI-estimated traffic/congestion level (clearly labeled
  as an estimate, not live data) plus a one-click "Open in Google Maps"
  link that shows ACTUAL live traffic in the user's browser for free.
- Safety advisories (theft-prone areas, slippery paths, etc.) are AI-
  generated general guidance based on common travel-safety knowledge —
  they are advisory only, not official/verified alerts.

NOTE on model names:
- Gemini model names get retired/renamed periodically by Google. This app
  tries a list of current model names in order and uses the first one
  that works, so a single retirement doesn't break the whole app.
"""

import os
import json
import re
import urllib.parse

import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="JourneyGenie — AI Tourist Guide",
    page_icon="🧭",
    layout="wide",
)

# --------------------------------------------------------------------------
# Sample fallback coordinates
# --------------------------------------------------------------------------
SAMPLE_COORDS = {
    "chennai": (13.0827, 80.2707),
    "mumbai": (19.0760, 72.8777),
    "delhi": (28.6139, 77.2090),
    "bengaluru": (12.9716, 77.5946),
    "bangalore": (12.9716, 77.5946),
    "goa": (15.2993, 74.1240),
    "jaipur": (26.9124, 75.7873),
    "kolkata": (22.5726, 88.3639),
    "hyderabad": (17.3850, 78.4867),
}

TRAFFIC_COLORS = {"Low": "🟢", "Moderate": "🟡", "Heavy": "🔴"}
SAFETY_COLORS = {"Safe": "🟢", "Caution": "🟡", "High Risk": "🔴"}

INTEREST_OPTIONS = [
    "Heritage", "Food", "Beaches", "Shopping", "Nightlife",
    "Adventure", "Nature & Wildlife", "Religious", "Museums", "Relaxation & Wellness",
]

# Models tried in order — first one that responds successfully is used.
# Update this list if Google retires/renames models again.
MODEL_CANDIDATES = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-flash-latest",
]


def get_city_center(destination: str):
    key = destination.strip().lower()
    return SAMPLE_COORDS.get(key, SAMPLE_COORDS["chennai"])


def google_maps_link(lat, lng, name):
    """Live-traffic-enabled Google Maps link (opens in browser, free, no API key)."""
    query = urllib.parse.quote(f"{name} @ {lat},{lng}")
    return f"https://www.google.com/maps/search/?api=1&query={query}"


# --------------------------------------------------------------------------
# Gemini itinerary generation
# --------------------------------------------------------------------------
PROMPT_TEMPLATE = """You are JourneyGenie, an expert local travel planner AI with
strong awareness of traveler safety.

Generate a personalized tourist itinerary as STRICT JSON ONLY (no markdown
fences, no commentary before or after) matching this exact schema:

{{
  "destination": "string",
  "total_estimated_cost": number,
  "stops": [
    {{
      "day": integer (1-indexed day number, must not exceed {time_days}),
      "time": "e.g. 09:00 AM",
      "name": "spot name",
      "category": "one of {interest_options_str}",
      "estimated_cost": number,
      "description": "1-2 sentence description",
      "lat": number,
      "lng": number,
      "traffic_level": "one of Low, Moderate, Heavy (typical congestion getting to this stop at this time)",
      "safety_level": "one of Safe, Caution, High Risk",
      "safety_note": "short advisory, e.g. 'Crowded market, watch belongings for pickpockets' or 'Rocks can be slippery near the shore' or 'No specific concerns' if Safe"
    }}
  ]
}}

Constraints:
- Destination: {destination}
- Total budget: INR {budget}
- Trip length: EXACTLY {time_days} day(s). You MUST generate stops covering all {time_days}
  day(s), roughly 3-5 stops per day, with the "day" field correctly numbered 1 through {time_days}.
- Interests (prioritize these): {interests}
- Sequence stops in a realistic time order within each day, with travel time buffers between stops.
- The sum of all "estimated_cost" values across the entire trip must not exceed the total budget.
- Provide realistic latitude/longitude coordinates for each stop within {destination}, India.
- Base safety_note on realistic, general common-knowledge travel-safety patterns for that
  type of location (e.g. crowded markets = pickpocket caution, wet rocks/waterfalls = slippery
  caution, isolated areas late at night = caution). Do not fabricate specific crime statistics.
- Return ONLY the JSON object, nothing else.
"""


def clean_json_response(raw_text: str) -> str:
    text = raw_text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def generate_itinerary(api_key, destination, budget, time_days, interests):
    if not GEMINI_AVAILABLE:
        raise RuntimeError(
            "google-generativeai is not installed. Run: pip install google-generativeai"
        )
    if not api_key:
        raise RuntimeError("Missing Gemini API key. Add it in the sidebar or as GEMINI_API_KEY.")

    genai.configure(api_key=api_key)

    prompt = PROMPT_TEMPLATE.format(
        destination=destination,
        budget=budget,
        time_days=time_days,
        interests=", ".join(interests) if interests else "general sightseeing",
        interest_options_str=", ".join(INTEREST_OPTIONS),
    )

    last_error = None
    for model_name in MODEL_CANDIDATES:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            raw = response.text
            cleaned = clean_json_response(raw)
            data = json.loads(cleaned)
            data["_model_used"] = model_name
            return data
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(
        f"All model candidates failed. Last error: {last_error}"
    )


# --------------------------------------------------------------------------
# Demo Mode — generates a full day-by-day itinerary for the exact
# number of days requested, without needing an API key.
# --------------------------------------------------------------------------
DEMO_STOP_TEMPLATES = [
    {"time": "09:00 AM", "name": "Heritage Museum", "category": "Heritage",
     "cost_frac": 0.02, "description": "Explore local history and culture.",
     "traffic_level": "Low", "safety_level": "Safe", "safety_note": "No specific concerns.",
     "d_lat": 0.010, "d_lng": 0.010},
    {"time": "11:00 AM", "name": "Street Food Market", "category": "Food",
     "cost_frac": 0.03, "description": "Sample authentic regional street food.",
     "traffic_level": "Heavy", "safety_level": "Caution",
     "safety_note": "Crowded market — keep an eye on bags and valuables (pickpocket risk).",
     "d_lat": -0.008, "d_lng": 0.015},
    {"time": "01:30 PM", "name": "Scenic Waterfront", "category": "Beaches",
     "cost_frac": 0.015, "description": "Relax by the water and enjoy the views.",
     "traffic_level": "Moderate", "safety_level": "Caution",
     "safety_note": "Rocks near the shoreline can be slippery — wear grippy footwear.",
     "d_lat": 0.020, "d_lng": -0.010},
    {"time": "04:00 PM", "name": "Central Shopping Street", "category": "Shopping",
     "cost_frac": 0.04, "description": "Browse local crafts and souvenirs.",
     "traffic_level": "Heavy", "safety_level": "Caution",
     "safety_note": "Busy pedestrian street — stay alert in dense crowds.",
     "d_lat": -0.015, "d_lng": -0.020},
    {"time": "07:30 PM", "name": "Rooftop Lounge", "category": "Nightlife",
     "cost_frac": 0.035, "description": "Unwind with live music and city views.",
     "traffic_level": "Moderate", "safety_level": "Safe", "safety_note": "No specific concerns.",
     "d_lat": 0.005, "d_lng": 0.020},
    {"time": "08:00 AM", "name": "Adventure Trail", "category": "Adventure",
     "cost_frac": 0.05, "description": "Guided outdoor adventure activity.",
     "traffic_level": "Low", "safety_level": "Caution",
     "safety_note": "Uneven terrain — wear proper footwear and stay with the group.",
     "d_lat": 0.030, "d_lng": 0.005},
    {"time": "10:00 AM", "name": "Wildlife Sanctuary", "category": "Nature & Wildlife",
     "cost_frac": 0.03, "description": "Spot local flora and fauna on a nature walk.",
     "traffic_level": "Low", "safety_level": "Safe", "safety_note": "No specific concerns.",
     "d_lat": -0.025, "d_lng": 0.012},
    {"time": "06:30 AM", "name": "Historic Temple", "category": "Religious",
     "cost_frac": 0.005, "description": "Visit a serene, historic place of worship.",
     "traffic_level": "Low", "safety_level": "Safe", "safety_note": "No specific concerns.",
     "d_lat": 0.008, "d_lng": -0.008},
    {"time": "02:00 PM", "name": "City Art Museum", "category": "Museums",
     "cost_frac": 0.015, "description": "Browse curated exhibits on regional art and history.",
     "traffic_level": "Moderate", "safety_level": "Safe", "safety_note": "No specific concerns.",
     "d_lat": -0.010, "d_lng": 0.005},
    {"time": "05:00 PM", "name": "Spa & Wellness Retreat", "category": "Relaxation & Wellness",
     "cost_frac": 0.06, "description": "Unwind with a relaxing spa session.",
     "traffic_level": "Low", "safety_level": "Safe", "safety_note": "No specific concerns.",
     "d_lat": 0.012, "d_lng": -0.015},
]


def get_demo_itinerary(destination, budget, time_days, interests):
    """Offline fallback that builds a full itinerary for the EXACT number
    of days requested, cycling through demo stop templates filtered by
    the selected interests (or all templates if none selected)."""
    lat, lng = get_city_center(destination)

    pool = [s for s in DEMO_STOP_TEMPLATES if not interests or s["category"] in interests]
    if not pool:
        pool = DEMO_STOP_TEMPLATES

    per_day_budget = budget / max(time_days, 1)
    stops = []
    pool_len = len(pool)

    for day in range(1, time_days + 1):
        # 3 stops per day, cycling through the filtered pool
        stops_today = min(3, pool_len) if pool_len > 0 else 0
        for i in range(stops_today):
            template = pool[(day - 1 + i) % pool_len]
            stop = dict(template)
            stop["day"] = day
            stop["name"] = f"{template['name']} ({destination})" if day == 1 else template["name"]
            stop["estimated_cost"] = round(per_day_budget * template["cost_frac"] * 6, 2)
            stop["lat"] = lat + template["d_lat"] * (1 + 0.1 * day)
            stop["lng"] = lng + template["d_lng"] * (1 + 0.1 * day)
            del stop["cost_frac"], stop["d_lat"], stop["d_lng"]
            stops.append(stop)

    total_cost = sum(s["estimated_cost"] for s in stops)
    # Scale down if demo total exceeds budget
    if total_cost > budget and total_cost > 0:
        scale = budget / total_cost
        for s in stops:
            s["estimated_cost"] = round(s["estimated_cost"] * scale, 2)
        total_cost = sum(s["estimated_cost"] for s in stops)

    return {"destination": destination, "total_estimated_cost": total_cost, "stops": stops}


# --------------------------------------------------------------------------
# Sidebar — user inputs
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧭 JourneyGenie")
    st.caption("AI Personalized Tourist Guide · Team SHE CODES · SIH 2026")
    st.divider()

    api_key_input = st.text_input(
        "Gemini API Key",
        value=os.environ.get("GEMINI_API_KEY", ""),
        type="password",
        help="Get a free key at https://ai.google.dev/. Leave blank to use Demo Mode.",
    )
    use_demo_mode = st.checkbox("Use Demo Mode (no API key needed)", value=not bool(api_key_input))

    st.divider()
    destination = st.text_input("📍 Destination / Location", value="Chennai")

    budget = st.slider(
        "💰 Budget (₹)",
        min_value=500,
        max_value=10_000_000,
        value=50_000,
        step=500,
        help="Up to ₹1 crore, for anything from a day trip to an extended multi-month journey.",
    )
    st.caption(f"Selected budget: ₹{budget:,.0f}")

    time_days = st.slider(
        "🗓️ Trip Length (days)",
        min_value=1,
        max_value=180,
        value=3,
        step=1,
        help="Up to 180 days (~6 months).",
    )
    if time_days >= 30:
        st.caption(f"Selected: {time_days} day(s) (~{time_days/30:.1f} months)")
    else:
        st.caption(f"Selected: {time_days} day(s)")

    interests = st.multiselect(
        "🎯 Interests",
        options=INTEREST_OPTIONS,
        default=["Heritage", "Food"],
    )

    st.divider()
    generate_clicked = st.button("🚀 Generate My Journey", use_container_width=True, type="primary")


# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------
st.title("🧭 JourneyGenie")
st.markdown(
    "##### AI-powered, budget-aware, multi-day travel itineraries with safety & traffic advisories — "
    "built for **Smart India Hackathon 2026** by **Team SHE CODES**"
)
st.divider()

if "itinerary" not in st.session_state:
    st.session_state.itinerary = None

if generate_clicked:
    with st.spinner(f"JourneyGenie is crafting your {time_days}-day personalized itinerary..."):
        try:
            if use_demo_mode or not api_key_input:
                itinerary = get_demo_itinerary(destination, budget, time_days, interests)
                st.session_state.used_demo = True
            else:
                itinerary = generate_itinerary(api_key_input, destination, budget, time_days, interests)
                st.session_state.used_demo = False
            st.session_state.itinerary = itinerary
        except Exception as e:
            st.error(f"Couldn't generate itinerary via Gemini ({e}). Falling back to Demo Mode.")
            st.session_state.itinerary = get_demo_itinerary(destination, budget, time_days, interests)
            st.session_state.used_demo = True

itinerary = st.session_state.itinerary

if itinerary is None:
    st.info("👈 Set your preferences in the sidebar and click **Generate My Journey** to get started.")
else:
    if st.session_state.get("used_demo"):
        st.warning("Showing a Demo Mode itinerary (no live Gemini call was made).", icon="🧪")
    elif itinerary.get("_model_used"):
        st.caption(f"✅ Generated live via Gemini model: `{itinerary['_model_used']}`")

    st.caption(
        "ℹ️ Traffic levels are AI-estimated typical congestion, not live sensor data. "
        "Safety notes are general AI advisory guidance, not official alerts. "
        "Use the 'Open in Google Maps' links on each stop for real live traffic."
    )

    stops = itinerary.get("stops", [])
    total_cost = itinerary.get("total_estimated_cost", sum(s.get("estimated_cost", 0) for s in stops))

    # ---------------- Budget metrics ----------------
    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Total Budget", f"₹{budget:,.0f}")
    col2.metric("🧾 Estimated Spend", f"₹{total_cost:,.0f}")
    remaining = budget - total_cost
    col3.metric("✅ Remaining", f"₹{remaining:,.0f}", delta=f"{(total_cost/budget)*100:.0f}% used" if budget else "0%")

    usage_pct = min(total_cost / budget, 1.0) if budget else 0
    st.progress(usage_pct, text=f"Budget usage: {usage_pct*100:.1f}%")

    st.divider()

    left, right = st.columns([1.2, 1])

    # ---------------- Timeline itinerary cards, grouped by day ----------------
    with left:
        st.subheader(f"🗓️ Your {time_days}-Day Itinerary")

        days_present = sorted(set(s.get("day", 1) for s in stops))
        for day_num in days_present:
            day_stops = [s for s in stops if s.get("day", 1) == day_num]
            st.markdown(f"### Day {day_num}")
            for stop in day_stops:
                traffic = stop.get("traffic_level", "Low")
                safety = stop.get("safety_level", "Safe")
                traffic_icon = TRAFFIC_COLORS.get(traffic, "🟢")
                safety_icon = SAFETY_COLORS.get(safety, "🟢")

                with st.container(border=True):
                    c1, c2 = st.columns([1, 3])
                    with c1:
                        st.markdown(f"**🕐 {stop.get('time', '-')}**")
                        st.caption(stop.get("category", ""))
                    with c2:
                        st.markdown(f"**{stop.get('name', 'Unnamed Spot')}**")
                        st.write(stop.get("description", ""))
                        st.markdown(f"💵 Estimated Cost: **₹{stop.get('estimated_cost', 0):,.0f}**")

                        badge_col1, badge_col2 = st.columns(2)
                        with badge_col1:
                            st.markdown(f"{traffic_icon} **Traffic:** {traffic}")
                        with badge_col2:
                            st.markdown(f"{safety_icon} **Safety:** {safety}")

                        if safety != "Safe":
                            st.warning(stop.get("safety_note", ""), icon="⚠️")

                        maps_url = google_maps_link(
                            stop.get("lat", get_city_center(destination)[0]),
                            stop.get("lng", get_city_center(destination)[1]),
                            stop.get("name", ""),
                        )
                        st.markdown(f"[🗺️ Open in Google Maps (live traffic)]({maps_url})")

    # ---------------- Interactive map ----------------
    with right:
        st.subheader("🗺️ Route Map")
        center_lat, center_lng = get_city_center(destination)
        if stops:
            center_lat = stops[0].get("lat", center_lat)
            center_lng = stops[0].get("lng", center_lng)

        route_map = folium.Map(location=[center_lat, center_lng], zoom_start=12, tiles="OpenStreetMap")

        coords = []
        for i, stop in enumerate(stops, start=1):
            lat = stop.get("lat", center_lat)
            lng = stop.get("lng", center_lng)
            coords.append((lat, lng))

            safety = stop.get("safety_level", "Safe")
            marker_color = "red" if safety == "High Risk" else ("orange" if safety == "Caution" else "blue")

            popup_html = (
                f"<b>Day {stop.get('day', 1)}: {stop.get('name', '')}</b><br/>"
                f"{stop.get('time', '')}<br/>"
                f"Traffic: {stop.get('traffic_level', 'Low')}<br/>"
                f"Safety: {safety}"
            )
            folium.Marker(
                location=[lat, lng],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"Day {stop.get('day', 1)}: {stop.get('name', '')}",
                icon=folium.Icon(color=marker_color, icon="info-sign"),
            ).add_to(route_map)

        if len(coords) > 1:
            folium.PolyLine(coords, color="#6c5ce7", weight=3, opacity=0.7).add_to(route_map)

        st_folium(route_map, width=None, height=420, returned_objects=[])
        st.caption("🔵 Safe   🟠 Caution   🔴 High Risk")

    st.divider()

    # ---------------- Budget breakdown table/chart ----------------
    st.subheader("📊 Budget Breakdown by Stop")
    if stops:
        df = pd.DataFrame(stops)
        display_cols = ["day", "name", "category", "estimated_cost", "traffic_level", "safety_level"]
        display_cols = [c for c in display_cols if c in df.columns]
        df_display = df[display_cols].rename(columns={
            "day": "Day", "name": "Spot", "category": "Category",
            "estimated_cost": "Estimated Cost (₹)",
            "traffic_level": "Traffic", "safety_level": "Safety",
        })
        b1, b2 = st.columns([1.2, 1])
        with b1:
            st.dataframe(df_display, use_container_width=True, hide_index=True)
        with b2:
            by_day = df.groupby("day")["estimated_cost"].sum() if "day" in df.columns else df.set_index("name")["estimated_cost"]
            st.bar_chart(by_day)

st.divider()
st.caption("JourneyGenie · Team SHE CODES · Smart India Hackathon 2026 · Problem Statement TC-S01")
