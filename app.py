"""
JourneyGenie — AI Personalized Tourist Guide
Team: SHE CODES | Smart India Hackathon 2026 | Problem Statement TC-S01

A single-file Streamlit MVP that uses the Google Gemini API to generate
a personalized, budget-aware, time-boxed travel itinerary, renders it as
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
      "time": "e.g. 09:00 AM",
      "name": "spot name",
      "category": "one of Heritage, Food, Beaches, Shopping, Nightlife",
      "estimated_cost": number,
      "description": "1-2 sentence description",
      "lat": number,
      "lng": number,
      "traffic_level": "one of Low, Moderate, Heavy (typical congestion getting to this stop at this time)",
      "safety_level": "one of Safe, Caution, High Risk",
      "safety_note": "short advisory, e.g. 'Crowded market, watch belongings for pickpockets' or 'Rocks can be slippery near the shore' or leave as 'No specific concerns' if Safe"
    }}
  ]
}}

Constraints:
- Destination: {destination}
- Total budget: INR {budget}
- Time available: {time_hours} hours ({time_days} day(s) covering multiple sub-itineraries if more than 24 hours)
- Interests (prioritize these): {interests}
- Sequence stops in a realistic time order, with travel time buffers between stops.
- If time_hours > 24, spread stops across multiple days, still reflected as a single time-ordered list
  with times like "Day 1, 09:00 AM", "Day 2, 10:00 AM" etc.
- The sum of all "estimated_cost" values must not exceed the total budget.
- Provide realistic latitude/longitude coordinates for each stop within {destination}, India.
- Include between 3 and 10 stops depending on how much time is available.
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


def generate_itinerary(api_key, destination, budget, time_hours, time_days, interests):
    if not GEMINI_AVAILABLE:
        raise RuntimeError(
            "google-generativeai is not installed. Run: pip install google-generativeai"
        )
    if not api_key:
        raise RuntimeError("Missing Gemini API key. Add it in the sidebar or as GEMINI_API_KEY.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash-lite")

    prompt = PROMPT_TEMPLATE.format(
        destination=destination,
        budget=budget,
        time_hours=time_hours,
        time_days=time_days,
        interests=", ".join(interests) if interests else "general sightseeing",
    )

    response = model.generate_content(prompt)
    raw = response.text
    cleaned = clean_json_response(raw)
    data = json.loads(cleaned)
    return data


def get_demo_itinerary(destination, budget, time_hours, time_days, interests):
    """Offline fallback so the app is demoable without an API key."""
    lat, lng = get_city_center(destination)
    demo_stops = [
        {"time": "Day 1, 09:00 AM", "name": f"{destination} Heritage Museum", "category": "Heritage",
         "estimated_cost": min(150, budget * 0.05), "description": "Explore local history and culture.",
         "lat": lat + 0.01, "lng": lng + 0.01,
         "traffic_level": "Low", "safety_level": "Safe", "safety_note": "No specific concerns."},
        {"time": "Day 1, 11:00 AM", "name": "Local Street Food Market", "category": "Food",
         "estimated_cost": min(400, budget * 0.1), "description": "Sample authentic regional street food.",
         "lat": lat - 0.008, "lng": lng + 0.015,
         "traffic_level": "Heavy", "safety_level": "Caution",
         "safety_note": "Crowded market — keep an eye on bags and valuables (pickpocket risk)."},
        {"time": "Day 1, 01:30 PM", "name": "Scenic Waterfront / Beach", "category": "Beaches",
         "estimated_cost": min(100, budget * 0.03), "description": "Relax by the water and enjoy the views.",
         "lat": lat + 0.02, "lng": lng - 0.01,
         "traffic_level": "Moderate", "safety_level": "Caution",
         "safety_note": "Rocks near the shoreline can be slippery — wear grippy footwear."},
        {"time": "Day 1, 04:00 PM", "name": "Central Shopping Street", "category": "Shopping",
         "estimated_cost": min(800, budget * 0.15), "description": "Browse local crafts and souvenirs.",
         "lat": lat - 0.015, "lng": lng - 0.02,
         "traffic_level": "Heavy", "safety_level": "Caution",
         "safety_note": "Busy pedestrian street — stay alert in dense crowds."},
        {"time": "Day 1, 07:30 PM", "name": "Rooftop Lounge", "category": "Nightlife",
         "estimated_cost": min(600, budget * 0.12), "description": "Unwind with live music and city views.",
         "lat": lat + 0.005, "lng": lng + 0.02,
         "traffic_level": "Moderate", "safety_level": "Safe", "safety_note": "No specific concerns."},
    ]
    filtered = [s for s in demo_stops if not interests or s["category"] in interests]
    if not filtered:
        filtered = demo_stops
    total_cost = sum(s["estimated_cost"] for s in filtered)
    return {"destination": destination, "total_estimated_cost": total_cost, "stops": filtered}


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
        max_value=100000,
        value=5000,
        step=500,
    )

    st.markdown("🕒 **Time Available**")
    time_days = st.slider("Days", min_value=0, max_value=7, value=0, step=1)
    time_hours_only = st.slider("Hours", min_value=0, max_value=23, value=6, step=1)
    time_hours = time_days * 24 + time_hours_only
    if time_hours == 0:
        time_hours = 2
    st.caption(f"Total: {time_hours} hours ({time_days} day(s) {time_hours_only} hour(s))")

    interests = st.multiselect(
        "🎯 Interests",
        options=["Heritage", "Food", "Beaches", "Shopping", "Nightlife"],
        default=["Heritage", "Food"],
    )

    st.divider()
    generate_clicked = st.button("🚀 Generate My Journey", use_container_width=True, type="primary")


# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------
st.title("🧭 JourneyGenie")
st.markdown(
    "##### AI-powered, budget-aware, time-boxed travel itineraries with safety & traffic advisories — "
    "built for **Smart India Hackathon 2026** by **Team SHE CODES**"
)
st.divider()

if "itinerary" not in st.session_state:
    st.session_state.itinerary = None

if generate_clicked:
    with st.spinner("JourneyGenie is crafting your personalized itinerary..."):
        try:
            if use_demo_mode or not api_key_input:
                itinerary = get_demo_itinerary(destination, budget, time_hours, time_days, interests)
                st.session_state.used_demo = True
            else:
                itinerary = generate_itinerary(api_key_input, destination, budget, time_hours, time_days, interests)
                st.session_state.used_demo = False
            st.session_state.itinerary = itinerary
        except Exception as e:
            st.error(f"Couldn't generate itinerary via Gemini ({e}). Falling back to Demo Mode.")
            st.session_state.itinerary = get_demo_itinerary(destination, budget, time_hours, time_days, interests)
            st.session_state.used_demo = True

itinerary = st.session_state.itinerary

if itinerary is None:
    st.info("👈 Set your preferences in the sidebar and click **Generate My Journey** to get started.")
else:
    if st.session_state.get("used_demo"):
        st.warning("Showing a Demo Mode itinerary (no live Gemini call was made).", icon="🧪")

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

    # ---------------- Timeline itinerary cards ----------------
    with left:
        st.subheader("🗓️ Your Itinerary")
        for stop in stops:
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

        route_map = folium.Map(location=[center_lat, center_lng], zoom_start=13, tiles="OpenStreetMap")

        coords = []
        for i, stop in enumerate(stops, start=1):
            lat = stop.get("lat", center_lat)
            lng = stop.get("lng", center_lng)
            coords.append((lat, lng))

            safety = stop.get("safety_level", "Safe")
            marker_color = "red" if safety == "High Risk" else ("orange" if safety == "Caution" else "blue")

            popup_html = (
                f"<b>{i}. {stop.get('name', '')}</b><br/>"
                f"{stop.get('time', '')}<br/>"
                f"Traffic: {stop.get('traffic_level', 'Low')}<br/>"
                f"Safety: {safety}"
            )
            folium.Marker(
                location=[lat, lng],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"{i}. {stop.get('name', '')}",
                icon=folium.Icon(color=marker_color, icon="info-sign"),
            ).add_to(route_map)

        if len(coords) > 1:
            folium.PolyLine(coords, color="#6c5ce7", weight=4, opacity=0.8).add_to(route_map)

        st_folium(route_map, width=None, height=420, returned_objects=[])
        st.caption("🔵 Safe   🟠 Caution   🔴 High Risk")

    st.divider()

    # ---------------- Budget breakdown table/chart ----------------
    st.subheader("📊 Budget Breakdown by Stop")
    if stops:
        df = pd.DataFrame(stops)
        display_cols = ["name", "category", "estimated_cost", "traffic_level", "safety_level"]
        display_cols = [c for c in display_cols if c in df.columns]
        df_display = df[display_cols].rename(columns={
            "name": "Spot", "category": "Category", "estimated_cost": "Estimated Cost (₹)",
            "traffic_level": "Traffic", "safety_level": "Safety",
        })
        b1, b2 = st.columns([1.2, 1])
        with b1:
            st.dataframe(df_display, use_container_width=True, hide_index=True)
        with b2:
            st.bar_chart(df.set_index("name")["estimated_cost"])

st.divider()
st.caption("JourneyGenie · Team SHE CODES · Smart India Hackathon 2026 · Problem Statement TC-S01")
