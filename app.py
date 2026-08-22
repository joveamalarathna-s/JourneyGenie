"""
JourneyGenie — AI Personalized Tourist Guide
Team: SHE CODES

A single-file Streamlit MVP that uses the Google Gemini API to generate a
personalized, budget-aware, multi-day travel itinerary for ANY destination
worldwide. Renders timeline cards with safety advisories, traffic estimates,
weather forecasts, emergency contacts, kid-friendly/dietary tags, and
suggested transport (cab/bus) options — plus an interactive map and budget
breakdown.

NOTES:
- Real-time traffic requires a paid Google Maps Traffic API. Instead, each
  stop gets an AI-estimated traffic/congestion level (clearly labeled as an
  estimate) plus a one-click "Open in Google Maps" link for real live traffic.
- Safety advisories are general AI guidance, not official/verified alerts.
- Weather uses the free Open-Meteo API (no key required), forecast-limited
  to ~16 days ahead, which is the practical limit of any weather forecast.
- Geocoding uses the free Open-Meteo Geocoding API (no key required) so any
  city worldwide can be searched, not just a fixed list.
- Gemini model names get retired/renamed periodically. This app tries a
  list of current model names in order and uses the first one that works.
"""

import os
import json
import re
import urllib.parse
import datetime

import pandas as pd
import requests
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

TRAFFIC_COLORS = {"Low": "🟢", "Moderate": "🟡", "Heavy": "🔴"}
SAFETY_COLORS = {"Safe": "🟢", "Caution": "🟡", "High Risk": "🔴"}
FOOD_ICONS = {"Veg": "🥦 Veg", "Non-Veg": "🍗 Non-Veg", "Both": "🍽️ Veg & Non-Veg", "N/A": ""}

INTEREST_OPTIONS = [
    "Heritage", "Food", "Beaches", "Shopping", "Nightlife",
    "Adventure", "Nature & Wildlife", "Religious", "Museums", "Relaxation & Wellness",
]

CURRENCY_OPTIONS = {
    "INR (₹)": "₹", "USD ($)": "$", "EUR (€)": "€", "GBP (£)": "£",
    "JPY (¥)": "¥", "AUD (A$)": "A$", "CAD (C$)": "C$", "SGD (S$)": "S$",
    "AED (د.إ)": "د.إ", "ZAR (R)": "R", "Other": "",
}

# Models tried in order — first one that responds successfully is used.
MODEL_CANDIDATES = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-flash-latest",
]

# Emergency numbers by country (Police / Ambulance / Fire where they differ).
# 112 is a near-universal GSM emergency number that works in most countries
# even without this list, so it's always shown as a fallback.
EMERGENCY_NUMBERS = {
    "india": {"Police": "100", "Ambulance": "108", "Fire": "101", "Women's Helpline": "1091", "Tourist Helpline": "1363"},
    "united states": {"Police/Ambulance/Fire": "911"},
    "united kingdom": {"Police/Ambulance/Fire": "999", "Non-emergency": "101"},
    "canada": {"Police/Ambulance/Fire": "911"},
    "australia": {"Police/Ambulance/Fire": "000"},
    "new zealand": {"Police/Ambulance/Fire": "111"},
    "japan": {"Police": "110", "Ambulance/Fire": "119"},
    "china": {"Police": "110", "Ambulance": "120", "Fire": "119"},
    "singapore": {"Police": "999", "Ambulance/Fire": "995"},
    "united arab emirates": {"Police": "999", "Ambulance": "998", "Fire": "997"},
    "germany": {"Police": "110", "Ambulance/Fire": "112"},
    "france": {"Police": "17", "Ambulance": "15", "Fire": "18", "EU-wide": "112"},
    "italy": {"Police": "113", "Ambulance": "118", "Fire": "115"},
    "spain": {"Police/Ambulance/Fire": "112"},
    "thailand": {"Police": "191", "Ambulance": "1669", "Tourist Police": "1155"},
    "indonesia": {"Police": "110", "Ambulance": "118", "Fire": "113"},
    "malaysia": {"Police/Ambulance/Fire": "999"},
    "south korea": {"Police": "112", "Ambulance/Fire": "119"},
    "brazil": {"Police": "190", "Ambulance": "192", "Fire": "193"},
    "mexico": {"Police/Ambulance/Fire": "911"},
    "south africa": {"Police": "10111", "Ambulance/Fire": "10177"},
    "russia": {"Police": "102", "Ambulance": "103", "Fire": "101"},
    "switzerland": {"Police": "117", "Ambulance": "144", "Fire": "118"},
    "netherlands": {"Police/Ambulance/Fire": "112"},
    "sri lanka": {"Police": "119", "Ambulance/Fire": "110"},
    "nepal": {"Police": "100", "Ambulance": "102", "Fire": "101"},
}

DEFAULT_COORDS = (28.6139, 77.2090)  # Delhi fallback if geocoding is unavailable


# --------------------------------------------------------------------------
# Geocoding (free, global, no API key) — Open-Meteo Geocoding API
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=86400)
def geocode_destination(name: str):
    """Returns (lat, lng, country, display_name). Falls back to Delhi on failure."""
    try:
        resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": name, "count": 1, "language": "en", "format": "json"},
            timeout=8,
        )
        data = resp.json()
        results = data.get("results")
        if results:
            r = results[0]
            display = r.get("name", name)
            if r.get("admin1"):
                display += f", {r['admin1']}"
            if r.get("country"):
                display += f", {r['country']}"
            return r["latitude"], r["longitude"], r.get("country", ""), display
    except Exception:
        pass
    return DEFAULT_COORDS[0], DEFAULT_COORDS[1], "", name


def get_emergency_numbers(country: str):
    key = (country or "").strip().lower()
    return EMERGENCY_NUMBERS.get(key)


# --------------------------------------------------------------------------
# Weather (free, no API key) — Open-Meteo Forecast API
# --------------------------------------------------------------------------
WEATHER_CODE_MAP = {
    0: ("☀️", "Clear sky"), 1: ("🌤️", "Mainly clear"), 2: ("⛅", "Partly cloudy"), 3: ("☁️", "Overcast"),
    45: ("🌫️", "Fog"), 48: ("🌫️", "Fog"),
    51: ("🌦️", "Light drizzle"), 53: ("🌦️", "Drizzle"), 55: ("🌧️", "Dense drizzle"),
    61: ("🌦️", "Light rain"), 63: ("🌧️", "Rain"), 65: ("🌧️", "Heavy rain"),
    71: ("🌨️", "Light snow"), 73: ("🌨️", "Snow"), 75: ("❄️", "Heavy snow"),
    80: ("🌦️", "Rain showers"), 81: ("🌧️", "Rain showers"), 82: ("⛈️", "Violent showers"),
    95: ("⛈️", "Thunderstorm"), 96: ("⛈️", "Thunderstorm w/ hail"), 99: ("⛈️", "Severe thunderstorm"),
}


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_weather(lat, lng, start_date_str, num_days):
    try:
        capped_days = min(num_days, 16)
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lng,
                "daily": "weathercode,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "timezone": "auto",
                "start_date": start_date_str,
                "end_date": (
                    datetime.date.fromisoformat(start_date_str) + datetime.timedelta(days=capped_days - 1)
                ).isoformat(),
            },
            timeout=8,
        )
        data = resp.json()
        daily = data.get("daily")
        if not daily:
            return None
        forecast = []
        for i, date_str in enumerate(daily["time"]):
            code = daily["weathercode"][i]
            icon, desc = WEATHER_CODE_MAP.get(code, ("🌡️", "Unknown"))
            forecast.append({
                "date": date_str,
                "icon": icon,
                "desc": desc,
                "temp_max": daily["temperature_2m_max"][i],
                "temp_min": daily["temperature_2m_min"][i],
                "rain_chance": daily.get("precipitation_probability_max", [None] * len(daily["time"]))[i],
            })
        return forecast
    except Exception:
        return None


# --------------------------------------------------------------------------
# Transport links
# --------------------------------------------------------------------------
def google_maps_link(lat, lng, name):
    query = urllib.parse.quote(f"{name} @ {lat},{lng}")
    return f"https://www.google.com/maps/search/?api=1&query={query}"


def google_maps_transit_link(dest_lat, dest_lng):
    d = urllib.parse.quote(f"{dest_lat},{dest_lng}")
    return f"https://www.google.com/maps/dir/?api=1&destination={d}&travelmode=transit"


def uber_deep_link(dest_lat, dest_lng, name):
    nickname = urllib.parse.quote(name)
    return (
        "https://m.uber.com/ul/?action=setPickup&pickup=my_location"
        f"&dropoff[latitude]={dest_lat}&dropoff[longitude]={dest_lng}&dropoff[nickname]={nickname}"
    )


# --------------------------------------------------------------------------
# Gemini itinerary generation
# --------------------------------------------------------------------------
PROMPT_TEMPLATE = """You are JourneyGenie, an expert local travel planner AI with
strong awareness of traveler safety, covering destinations anywhere in the world.

Generate a personalized tourist itinerary as STRICT JSON ONLY (no markdown
fences, no commentary before or after) matching this exact schema:

{{
  "destination": "string",
  "country": "string",
  "total_estimated_cost": number,
  "stops": [
    {{
      "day": integer (1-indexed day number, must not exceed {time_days}),
      "time": "e.g. 09:00 AM",
      "name": "spot name",
      "category": "one of {interest_options_str}",
      "estimated_cost": number (in {currency_code}),
      "description": "1-2 sentence description",
      "lat": number,
      "lng": number,
      "traffic_level": "one of Low, Moderate, Heavy",
      "safety_level": "one of Safe, Caution, High Risk",
      "safety_note": "short advisory or 'No specific concerns' if Safe",
      "kid_friendly": true or false,
      "food_type": "one of Veg, Non-Veg, Both, N/A (N/A for non-Food stops)",
      "transport_mode": "best suited mode to reach this stop: one of Cab, Bus, Metro, Walk, Auto-rickshaw, Train",
      "transport_cost": number (approx one-way fare in {currency_code}),
      "transport_notes": "short note, e.g. 'Bus route or line name, ~15 min' or 'Short walk from previous stop'"
    }}
  ]
}}

Constraints:
- Destination: {destination} (location context: {country_hint})
- Total budget: {currency_code} {budget}
- Trip length: EXACTLY {time_days} day(s). Generate stops covering all {time_days}
  day(s), roughly 3-5 stops per day, with "day" numbered 1 through {time_days}.
- Interests (prioritize these): {interests}
- Sequence stops in a realistic time order within each day, with travel buffers.
- The sum of all "estimated_cost" values across the trip must not exceed the budget.
- Provide realistic latitude/longitude coordinates for each stop near {destination}.
- Base safety_note on realistic, general common-knowledge travel-safety patterns
  (e.g. crowded markets = pickpocket caution, wet rocks = slippery caution). Do not
  fabricate specific crime statistics.
- {kids_mode_instruction}
- {dietary_instruction}
- Return ONLY the JSON object, nothing else.
"""


def clean_json_response(raw_text: str) -> str:
    text = raw_text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def generate_itinerary(api_key, destination, country_hint, budget, time_days, interests,
                        kids_mode, dietary_pref, currency_code):
    if not GEMINI_AVAILABLE:
        raise RuntimeError("google-generativeai is not installed. Run: pip install google-generativeai")
    if not api_key:
        raise RuntimeError("Missing Gemini API key. Add it in the sidebar or as GEMINI_API_KEY.")

    genai.configure(api_key=api_key)

    kids_mode_instruction = (
        "KIDS MODE IS ON: every stop must have kid_friendly = true, avoid nightlife and adult-oriented "
        "venues entirely, and prefer safety_level Safe over Caution/High Risk wherever possible."
        if kids_mode else
        "Kids Mode is off — include a natural mix of stops for a general traveler."
    )
    dietary_instruction = (
        "Dietary preference: Vegetarian only. Every Food-category stop must have food_type = 'Veg'."
        if dietary_pref == "Vegetarian only" else
        "Dietary preference: no restriction — Food stops can be Veg, Non-Veg, or Both."
    )

    prompt = PROMPT_TEMPLATE.format(
        destination=destination,
        country_hint=country_hint or "unspecified",
        budget=budget,
        time_days=time_days,
        interests=", ".join(interests) if interests else "general sightseeing",
        interest_options_str=", ".join(INTEREST_OPTIONS),
        currency_code=currency_code,
        kids_mode_instruction=kids_mode_instruction,
        dietary_instruction=dietary_instruction,
    )

    last_error = None
    for model_name in MODEL_CANDIDATES:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            cleaned = clean_json_response(response.text)
            data = json.loads(cleaned)
            data["_model_used"] = model_name
            return data
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"All model candidates failed. Last error: {last_error}")


# --------------------------------------------------------------------------
# Demo Mode — full day-by-day itinerary without needing an API key
# --------------------------------------------------------------------------
DEMO_STOP_TEMPLATES = [
    {"time": "09:00 AM", "name": "Heritage Museum", "category": "Heritage", "cost_frac": 0.02,
     "description": "Explore local history and culture.", "traffic_level": "Low", "safety_level": "Safe",
     "safety_note": "No specific concerns.", "kid_friendly": True, "food_type": "N/A",
     "transport_mode": "Cab", "transport_cost_frac": 0.004, "transport_notes": "~10 min cab ride.",
     "d_lat": 0.010, "d_lng": 0.010},
    {"time": "11:00 AM", "name": "Street Food Market", "category": "Food", "cost_frac": 0.03,
     "description": "Sample authentic regional street food.", "traffic_level": "Heavy", "safety_level": "Caution",
     "safety_note": "Crowded market — keep an eye on bags and valuables (pickpocket risk).",
     "kid_friendly": True, "food_type": "Both",
     "transport_mode": "Bus", "transport_cost_frac": 0.001, "transport_notes": "Local bus, ~15 min.",
     "d_lat": -0.008, "d_lng": 0.015},
    {"time": "12:30 PM", "name": "Vegetarian Thali House", "category": "Food", "cost_frac": 0.025,
     "description": "Traditional multi-course vegetarian meal.", "traffic_level": "Moderate", "safety_level": "Safe",
     "safety_note": "No specific concerns.", "kid_friendly": True, "food_type": "Veg",
     "transport_mode": "Walk", "transport_cost_frac": 0.0, "transport_notes": "Short walk from previous stop.",
     "d_lat": 0.004, "d_lng": -0.006},
    {"time": "01:30 PM", "name": "Scenic Waterfront", "category": "Beaches", "cost_frac": 0.015,
     "description": "Relax by the water and enjoy the views.", "traffic_level": "Moderate", "safety_level": "Caution",
     "safety_note": "Rocks near the shoreline can be slippery — wear grippy footwear.",
     "kid_friendly": True, "food_type": "N/A",
     "transport_mode": "Cab", "transport_cost_frac": 0.005, "transport_notes": "~12 min cab ride.",
     "d_lat": 0.020, "d_lng": -0.010},
    {"time": "04:00 PM", "name": "Central Shopping Street", "category": "Shopping", "cost_frac": 0.04,
     "description": "Browse local crafts and souvenirs.", "traffic_level": "Heavy", "safety_level": "Caution",
     "safety_note": "Busy pedestrian street — stay alert in dense crowds.",
     "kid_friendly": True, "food_type": "N/A",
     "transport_mode": "Metro", "transport_cost_frac": 0.002, "transport_notes": "Nearest metro station, ~8 min.",
     "d_lat": -0.015, "d_lng": -0.020},
    {"time": "07:30 PM", "name": "Rooftop Lounge", "category": "Nightlife", "cost_frac": 0.035,
     "description": "Unwind with live music and city views.", "traffic_level": "Moderate", "safety_level": "Safe",
     "safety_note": "No specific concerns.", "kid_friendly": False, "food_type": "Both",
     "transport_mode": "Cab", "transport_cost_frac": 0.006, "transport_notes": "~15 min cab ride, evening traffic.",
     "d_lat": 0.005, "d_lng": 0.020},
    {"time": "08:00 AM", "name": "Adventure Trail", "category": "Adventure", "cost_frac": 0.05,
     "description": "Guided outdoor adventure activity.", "traffic_level": "Low", "safety_level": "Caution",
     "safety_note": "Uneven terrain — wear proper footwear and stay with the group.",
     "kid_friendly": False, "food_type": "N/A",
     "transport_mode": "Cab", "transport_cost_frac": 0.008, "transport_notes": "~25 min cab ride out of town.",
     "d_lat": 0.030, "d_lng": 0.005},
    {"time": "10:00 AM", "name": "Wildlife Sanctuary", "category": "Nature & Wildlife", "cost_frac": 0.03,
     "description": "Spot local flora and fauna on a nature walk.", "traffic_level": "Low", "safety_level": "Safe",
     "safety_note": "No specific concerns.", "kid_friendly": True, "food_type": "N/A",
     "transport_mode": "Bus", "transport_cost_frac": 0.003, "transport_notes": "Tourist shuttle bus, ~20 min.",
     "d_lat": -0.025, "d_lng": 0.012},
    {"time": "06:30 AM", "name": "Historic Temple", "category": "Religious", "cost_frac": 0.005,
     "description": "Visit a serene, historic place of worship.", "traffic_level": "Low", "safety_level": "Safe",
     "safety_note": "No specific concerns.", "kid_friendly": True, "food_type": "N/A",
     "transport_mode": "Walk", "transport_cost_frac": 0.0, "transport_notes": "Walkable from most central areas.",
     "d_lat": 0.008, "d_lng": -0.008},
    {"time": "02:00 PM", "name": "City Art Museum", "category": "Museums", "cost_frac": 0.015,
     "description": "Browse curated exhibits on regional art and history.", "traffic_level": "Moderate",
     "safety_level": "Safe", "safety_note": "No specific concerns.", "kid_friendly": True, "food_type": "N/A",
     "transport_mode": "Metro", "transport_cost_frac": 0.002, "transport_notes": "Metro, ~10 min.",
     "d_lat": -0.010, "d_lng": 0.005},
    {"time": "05:00 PM", "name": "Spa & Wellness Retreat", "category": "Relaxation & Wellness", "cost_frac": 0.06,
     "description": "Unwind with a relaxing spa session.", "traffic_level": "Low", "safety_level": "Safe",
     "safety_note": "No specific concerns.", "kid_friendly": False, "food_type": "N/A",
     "transport_mode": "Cab", "transport_cost_frac": 0.004, "transport_notes": "~10 min cab ride.",
     "d_lat": 0.012, "d_lng": -0.015},
]


def get_demo_itinerary(destination, lat, lng, budget, time_days, interests, kids_mode, dietary_pref):
    pool = [s for s in DEMO_STOP_TEMPLATES if not interests or s["category"] in interests]
    if kids_mode:
        pool = [s for s in pool if s["kid_friendly"]]
    if dietary_pref == "Vegetarian only":
        pool = [s for s in pool if s["category"] != "Food" or s["food_type"] in ("Veg", "Both")]
    if not pool:
        pool = DEMO_STOP_TEMPLATES

    per_day_budget = budget / max(time_days, 1)
    stops = []
    pool_len = len(pool)

    for day in range(1, time_days + 1):
        stops_today = min(3, pool_len) if pool_len > 0 else 0
        for i in range(stops_today):
            template = pool[(day - 1 + i) % pool_len]
            stop = dict(template)
            stop["day"] = day
            stop["name"] = f"{template['name']} ({destination})" if day == 1 else template["name"]
            stop["estimated_cost"] = round(per_day_budget * template["cost_frac"] * 6, 2)
            stop["transport_cost"] = round(per_day_budget * template["transport_cost_frac"] * 6, 2)
            stop["lat"] = lat + template["d_lat"] * (1 + 0.1 * day)
            stop["lng"] = lng + template["d_lng"] * (1 + 0.1 * day)
            for k in ("cost_frac", "transport_cost_frac", "d_lat", "d_lng"):
                del stop[k]
            stops.append(stop)

    total_cost = sum(s["estimated_cost"] for s in stops)
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
    st.caption("AI Personalized Tourist Guide · Team SHE CODES")
    st.divider()

    api_key_input = os.environ.get("GEMINI_API_KEY", "")
    use_demo_mode = st.checkbox("Use Demo Mode (no live AI call)", value=not bool(api_key_input))
    if api_key_input:
        st.caption("🔑 Gemini API key loaded securely from app settings.")
    else:
        st.caption("⚠️ No Gemini API key configured — running in Demo Mode.")

    st.divider()
    destination = st.text_input("📍 Destination (any city, worldwide)", value="Chennai, India")

    currency_label = st.selectbox("💱 Currency", options=list(CURRENCY_OPTIONS.keys()), index=0)
    currency_symbol = CURRENCY_OPTIONS[currency_label]
    currency_code = currency_label.split(" ")[0]

    budget = st.number_input(
        f"💰 Budget ({currency_symbol or currency_code})",
        min_value=0, value=50000, step=500,
        help="Type your exact budget — no upper limit.",
    )
    st.caption(f"Selected budget: {currency_symbol}{budget:,.0f}")

    st.markdown("🗓️ **Travel Dates**")
    today = datetime.date.today()
    date_col1, date_col2 = st.columns(2)
    with date_col1:
        start_date = st.date_input("Start date", value=today, min_value=today)
    with date_col2:
        end_date = st.date_input("End date", value=today + datetime.timedelta(days=2), min_value=start_date)

    if end_date < start_date:
        st.error("End date must be on or after the start date.")
        time_days = 1
    else:
        time_days = (end_date - start_date).days + 1
    st.caption(f"Trip length: {time_days} day(s)" + (f" (~{time_days/30:.1f} months)" if time_days >= 30 else ""))

    interests = st.multiselect("🎯 Interests", options=INTEREST_OPTIONS, default=["Heritage", "Food"])

    st.divider()
    st.markdown("👶 **Family & Dietary**")
    kids_mode = st.checkbox("Kids Mode (family-friendly stops only)", value=False)
    dietary_pref = st.selectbox("Dietary preference", options=["No preference", "Vegetarian only"])

    st.divider()
    st.markdown("🚨 **Emergency Contact (optional)**")
    contact_name = st.text_input("Contact name", placeholder="e.g. Mom, Dad, a friend")
    contact_phone = st.text_input("Contact phone number", placeholder="e.g. +91 98765 43210")

    st.divider()
    generate_clicked = st.button("🚀 Generate My Journey", use_container_width=True, type="primary")


# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------
st.title("🧭 JourneyGenie")
st.markdown(
    "##### AI-powered, budget-aware, multi-day travel itineraries with safety, weather, "
    "and transport guidance — for destinations anywhere in the world, by **Team SHE CODES**"
)
st.divider()

if "itinerary" not in st.session_state:
    st.session_state.itinerary = None

if generate_clicked:
    with st.spinner(f"JourneyGenie is crafting your {time_days}-day itinerary for {destination}..."):
        lat, lng, country, display_name = geocode_destination(destination)
        st.session_state.geo = (lat, lng, country, display_name)
        try:
            if use_demo_mode or not api_key_input:
                itinerary = get_demo_itinerary(destination, lat, lng, budget, time_days, interests, kids_mode, dietary_pref)
                st.session_state.used_demo = True
            else:
                itinerary = generate_itinerary(
                    api_key_input, destination, country, budget, time_days, interests,
                    kids_mode, dietary_pref, currency_code,
                )
                st.session_state.used_demo = False
            st.session_state.itinerary = itinerary
        except Exception as e:
            st.error(f"Couldn't generate itinerary via Gemini ({e}). Falling back to Demo Mode.")
            st.session_state.itinerary = get_demo_itinerary(destination, lat, lng, budget, time_days, interests, kids_mode, dietary_pref)
            st.session_state.used_demo = True

itinerary = st.session_state.itinerary

if itinerary is None:
    st.info("👈 Set your preferences in the sidebar and click **Generate My Journey** to get started.")
else:
    geo_lat, geo_lng, geo_country, geo_display = st.session_state.get("geo", (*DEFAULT_COORDS, "", destination))

    if st.session_state.get("used_demo"):
        st.warning("Showing a Demo Mode itinerary (no live Gemini call was made).", icon="🧪")
    elif itinerary.get("_model_used"):
        st.caption(f"✅ Generated live via Gemini model: `{itinerary['_model_used']}`")

    st.caption(
        "ℹ️ Traffic levels and transport fares are AI-estimated, not live data. Safety notes are general "
        "AI advisory guidance, not official alerts. Weather is a live forecast (up to 16 days ahead)."
    )

    # ---------------- Emergency contacts panel ----------------
    with st.expander("🚨 Safety & Emergency Contacts", expanded=True):
        e1, e2 = st.columns(2)
        with e1:
            st.markdown(f"**Destination country:** {geo_country or 'Unknown'}")
            numbers = get_emergency_numbers(geo_country)
            if numbers:
                for label, num in numbers.items():
                    st.markdown(f"- **{label}:** {num}")
            else:
                st.markdown("- No specific listing available for this country.")
            st.markdown("- **Universal fallback:** dial **112** (works on most mobile networks worldwide).")
        with e2:
            if contact_name and contact_phone:
                st.markdown("**Your personal emergency contact:**")
                st.markdown(f"- {contact_name}: **{contact_phone}**")
            else:
                st.caption("Add a personal emergency contact in the sidebar to see it here.")

    # ---------------- Weather forecast ----------------
    st.subheader("☀️ Weather Forecast")
    forecast = fetch_weather(geo_lat, geo_lng, start_date.isoformat(), time_days)
    if forecast:
        if time_days > 16:
            st.caption(f"Showing the first 16 of {time_days} days — forecasts beyond that aren't reliable yet.")
        w_cols = st.columns(min(len(forecast), 8))
        for i, day_w in enumerate(forecast[:8]):
            with w_cols[i % 8]:
                st.markdown(f"**{day_w['date'][5:]}**")
                st.markdown(f"{day_w['icon']}")
                st.caption(day_w["desc"])
                st.markdown(f"{day_w['temp_min']:.0f}° – {day_w['temp_max']:.0f}°C")
                if day_w.get("rain_chance") is not None:
                    st.caption(f"🌧️ {day_w['rain_chance']}%")
        if len(forecast) > 8:
            st.caption(f"+ {len(forecast) - 8} more day(s) in the forecast window.")
    else:
        st.caption("Weather forecast unavailable right now — check your connection or try again shortly.")

    st.divider()

    stops = itinerary.get("stops", [])
    total_cost = itinerary.get("total_estimated_cost", sum(s.get("estimated_cost", 0) for s in stops))

    # ---------------- Budget metrics ----------------
    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Total Budget", f"{currency_symbol}{budget:,.0f}")
    col2.metric("🧾 Estimated Spend", f"{currency_symbol}{total_cost:,.0f}")
    remaining = budget - total_cost
    col3.metric("✅ Remaining", f"{currency_symbol}{remaining:,.0f}",
                delta=f"{(total_cost/budget)*100:.0f}% used" if budget else "0%")
    usage_pct = min(total_cost / budget, 1.0) if budget else 0
    st.progress(usage_pct, text=f"Budget usage: {usage_pct*100:.1f}%")

    st.divider()
    left, right = st.columns([1.2, 1])

    # ---------------- Timeline itinerary cards, grouped by day ----------------
    with left:
        st.subheader(
            f"🗓️ Your {time_days}-Day Itinerary — {geo_display} "
            f"({start_date.strftime('%d %b %Y')} – {end_date.strftime('%d %b %Y')})"
        )

        days_present = sorted(set(s.get("day", 1) for s in stops))
        for day_num in days_present:
            day_stops = [s for s in stops if s.get("day", 1) == day_num]
            actual_date = start_date + datetime.timedelta(days=day_num - 1)
            st.markdown(f"### Day {day_num} — {actual_date.strftime('%A, %d %b %Y')}")

            for stop in day_stops:
                traffic = stop.get("traffic_level", "Low")
                safety = stop.get("safety_level", "Safe")
                traffic_icon = TRAFFIC_COLORS.get(traffic, "🟢")
                safety_icon = SAFETY_COLORS.get(safety, "🟢")
                kid_badge = "👶 Kid-Friendly" if stop.get("kid_friendly") else "🚫 Not Ideal for Kids"
                food_badge = FOOD_ICONS.get(stop.get("food_type", "N/A"), "")

                with st.container(border=True):
                    c1, c2 = st.columns([1, 3])
                    with c1:
                        st.markdown(f"**🕐 {stop.get('time', '-')}**")
                        st.caption(stop.get("category", ""))
                    with c2:
                        st.markdown(f"**{stop.get('name', 'Unnamed Spot')}**")
                        st.write(stop.get("description", ""))
                        st.markdown(f"💵 Estimated Cost: **{currency_symbol}{stop.get('estimated_cost', 0):,.0f}**")

                        badge_col1, badge_col2, badge_col3 = st.columns(3)
                        with badge_col1:
                            st.markdown(f"{traffic_icon} **Traffic:** {traffic}")
                        with badge_col2:
                            st.markdown(f"{safety_icon} **Safety:** {safety}")
                        with badge_col3:
                            st.markdown(kid_badge)
                        if food_badge:
                            st.markdown(food_badge)

                        if safety != "Safe":
                            st.warning(stop.get("safety_note", ""), icon="⚠️")

                        transport_mode = stop.get("transport_mode", "Cab")
                        transport_cost = stop.get("transport_cost", 0)
                        transport_notes = stop.get("transport_notes", "")
                        st.markdown(
                            f"🚗 **Suggested transport:** {transport_mode} "
                            f"(~{currency_symbol}{transport_cost:,.0f}) — {transport_notes}"
                        )

                        s_lat = stop.get("lat", geo_lat)
                        s_lng = stop.get("lng", geo_lng)
                        link_col1, link_col2, link_col3 = st.columns(3)
                        with link_col1:
                            st.markdown(f"[🗺️ Maps]({google_maps_link(s_lat, s_lng, stop.get('name', ''))})")
                        with link_col2:
                            st.markdown(f"[🚌 Bus/Transit]({google_maps_transit_link(s_lat, s_lng)})")
                        with link_col3:
                            st.markdown(f"[🚕 Book Cab]({uber_deep_link(s_lat, s_lng, stop.get('name', ''))})")

    # ---------------- Interactive map ----------------
    with right:
        st.subheader("🗺️ Route Map")
        center_lat, center_lng = geo_lat, geo_lng
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
                f"{stop.get('time', '')}<br/>Traffic: {stop.get('traffic_level', 'Low')}<br/>"
                f"Safety: {safety}<br/>Transport: {stop.get('transport_mode', '')}"
            )
            folium.Marker(
                location=[lat, lng], popup=folium.Popup(popup_html, max_width=250),
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
        display_cols = ["day", "name", "category", "estimated_cost", "transport_mode",
                         "transport_cost", "traffic_level", "safety_level", "kid_friendly", "food_type"]
        display_cols = [c for c in display_cols if c in df.columns]
        df_display = df[display_cols].rename(columns={
            "day": "Day", "name": "Spot", "category": "Category",
            "estimated_cost": f"Cost ({currency_symbol})", "transport_mode": "Transport",
            "transport_cost": f"Transport Cost ({currency_symbol})",
            "traffic_level": "Traffic", "safety_level": "Safety",
            "kid_friendly": "Kid-Friendly", "food_type": "Food Type",
        })
        b1, b2 = st.columns([1.2, 1])
        with b1:
            st.dataframe(df_display, use_container_width=True, hide_index=True)
        with b2:
            by_day = df.groupby("day")["estimated_cost"].sum() if "day" in df.columns else df.set_index("name")["estimated_cost"]
            st.bar_chart(by_day)

st.divider()
st.caption("JourneyGenie · Team SHE CODES")
