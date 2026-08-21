# 🧭 JourneyGenie — AI Personalized Tourist Guide

<p align="center">
  <img src="https://img.shields.io/badge/SIH-2026-orange?style=for-the-badge" alt="SIH 2026"/>
  <img src="https://img.shields.io/badge/Team-SHE%20CODES-purple?style=for-the-badge" alt="Team SHE CODES"/>
  <img src="https://img.shields.io/badge/PS%20ID-TC--S01-blue?style=for-the-badge" alt="Problem Statement ID"/>
  <img src="https://img.shields.io/badge/Event-Blaze%20a%20Trail%204.0-green?style=for-the-badge" alt="Blaze a Trail 4.0"/>
</p>

<p align="center">
  <b>Smart India Hackathon 2026</b> · Blaze a Trail 4.0, St. Joseph's Institute of Technology<br/>
  <b>Problem Statement:</b> TC-S01 — AI Personalized Tourist Guide Software<br/>
  <b>Team:</b> SHE CODES
</p>

---

## 📌 Abstract

Traditional trip-planning tools give every traveler the same generic checklist of "top attractions," ignoring individual budgets, available time, and personal interests. **JourneyGenie** is an AI travel assistant that generates a **hyper-personalized, budget-aware, time-boxed itinerary** in seconds — combining a large language model (Google Gemini) with structured constraint solving, geospatial mapping, and real-time re-routing logic.

Given a destination, a budget range, available hours, and a set of interests (Heritage, Food, Beaches, Shopping, Nightlife), JourneyGenie produces a spot-by-spot timeline — complete with estimated costs, descriptions, and an interactive map — so a tourist can go from "I have 6 hours and ₹2,000 in Chennai" to a ready-to-follow plan instantly.

## 🏗️ System Architecture
┌─────────────────┐ ┌──────────────────────┐ ┌────────────────────┐
│ Streamlit UI │─────▶│ Constraint Layer │─────▶│ Gemini LLM Engine │
│ (Sidebar Inputs) │ │ (Budget/Time/Interest) │ │ (Prompt + Schema) │
└─────────────────┘ └──────────────────────┘ └────────────────────┘
│ │
│ ▼
│ ┌────────────────────────┐
│ │ Structured JSON Plan │
│ │ (spots, cost, time, geo)│
│ └────────────────────────┘
▼ │
┌─────────────────┐ ┌──────────────────────┐ │
│ Timeline Cards │◀─────│ Response Parser │◀──────────────┘
│ (Itinerary View) │ │ + Budget Aggregator │
└─────────────────┘ └──────────────────────┘
│
▼
┌─────────────────────────────────────────────┐
│ Folium Interactive Map (Route + Markers) │
│ + Vector-Search-ready POI store (MongoDB) │
└─────────────────────────────────────────────┘


**Flow:** User constraints → prompt engineered against Gemini with a strict JSON schema → response validated/parsed → rendered as timeline cards, budget metrics, and an interactive Folium map → re-routing loop re-queries the model when a spot is skipped or time/budget changes mid-trip.

## ✨ Core Features

| Feature | Description |
|---|---|
| 🎯 **Hyper-Personalization** | Itineraries are generated per-user from a weighted interest vector (Heritage, Food, Beaches, Shopping, Nightlife) rather than static "top 10" lists. |
| 💰 **Budget-Aware Optimization** | Every recommended spot carries an estimated cost; the planner greedily fits activities so total spend stays within the user's slider-selected budget, with a live usage breakdown. |
| 🔄 **Real-Time Re-routing** | If a user skips a stop, runs late, or a location closes, the assistant re-plans the remaining timeline on the fly, preserving budget and interest constraints. |
| 🗺️ **Geospatial Visualization** | Every itinerary is plotted on an interactive Folium map with route markers, so the plan is spatially intuitive, not just a list. |
| 🕒 **Time-Boxed Planning** | Plans respect a strict available-hours window (2–24 hrs), sequencing spots realistically with travel buffers. |

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | Streamlit (MVP) / React (planned production UI) | Interactive web UI for input collection and itinerary display |
| **AI / Orchestration** | Google Gemini API, LangChain | Natural-language itinerary generation, prompt chaining, structured output |
| **Data / Retrieval** | Vector Search (embeddings) + MongoDB | Semantic POI (point-of-interest) retrieval and persistence at scale |
| **Maps & Geospatial** | Folium, Google Maps API | Route visualization, geocoding, distance/time estimation |
| **Core Language** | Python 3.10+ | Application logic, data processing |
| **Data Handling** | Pandas | Itinerary tabulation and budget aggregation |

## 📊 Feasibility, Viability & Risk Mitigation

| Aspect | Assessment | Mitigation Strategy |
|---|---|---|
| **Technical Feasibility** | High — Gemini API, Streamlit, and Folium are mature, well-documented, and require no custom model training for the MVP. | Use structured JSON-mode prompting to keep LLM output reliable and parseable. |
| **LLM Hallucination Risk** | Medium — the model may suggest non-existent or closed venues. | Cross-validate suggested spots against a curated/vector-search POI database before display; add a "verify on map" fallback. |
| **API Cost & Rate Limits** | Medium — Gemini API calls scale with users. | Cache common itineraries, batch prompts, and implement request throttling; fall back to a lightweight rule-based planner if quota is exceeded. |
| **Budget Accuracy** | Medium — real-world prices fluctuate. | Treat costs as estimates, refreshed periodically via a pricing dataset; clearly label them "approximate." |
| **Market Viability** | High — India's domestic tourism sector is growing rapidly and lacks affordable AI-personalized planning tools. | Position as a freemium travel-tech product; monetize via premium re-routing, offline maps, and partner bookings. |
| **Scalability** | High — stateless Streamlit MVP can be containerized and horizontally scaled; MongoDB + vector search scale independently of the LLM layer. | Migrate to a microservice architecture (FastAPI backend + React frontend) post-MVP. |
| **Data Privacy** | Medium — location and spending data are sensitive. | No persistent storage of personal data in the MVP; anonymized session-only state. |

## 🚀 Setup & Quickstart Guide

### Prerequisites
- Python 3.10 or higher
- A Google Gemini API key ([get one here](https://ai.google.dev/))

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-org>/JourneyGenie.git
cd JourneyGenie

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Gemini API key
export GEMINI_API_KEY="your-api-key-here"     # On Windows (PowerShell): $env:GEMINI_API_KEY="your-api-key-here"
```

### Run the App

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`. Enter your Gemini API key in the sidebar if it isn't picked up from the environment, set your destination, budget, time, and interests, then click **Generate My Journey** 🚀.

## 📚 Research Citations & References

1. **Ministry of Tourism, Government of India** — *India Tourism Statistics 2024*, highlighting domestic tourist visit growth and the demand gap for personalized digital travel planning tools.
2. **Ministry of Tourism, Government of India** — *National Strategy for Sustainable Tourism*, emphasizing AI-driven, low-footprint itinerary planning to distribute tourist load across lesser-known destinations.
3. Gavalas, D., Konstantopoulos, C., Mastakas, K., & Pantziou, G. (2014). *Mobile recommender systems in tourism.* Journal of Network and Computer Applications — foundational survey on constraint-based tourist trip design problems (TTDP).
4. Vansteenwegen, P., & Van Oudheusden, D. (2007). *The Orienteering Problem: A survey.* European Journal of Operational Research — theoretical basis for time/budget-constrained route optimization used in itinerary sequencing.
5. IEEE Access / IEEE Transactions papers on **LLM-based conversational travel recommendation systems**, informing the prompt-engineering and structured-output approach used for Gemini integration in this project.

*(Full citation list with DOIs to be appended in `/docs/references.md` for the final submission.)*

## 📁 Repository Structure

JourneyGenie/
├── app.py # Streamlit MVP application
├── requirements.txt # Python dependencies
├── README.md # This file
├── .gitignore
└── docs/ # (optional) architecture diagrams, PPT, references


## 👩‍💻 Team SHE CODES

Built with ❤️ for Smart India Hackathon 2026 — Blaze a Trail 4.0, St. Joseph's Institute of Technology.

## 📄 License

This project is released under the MIT License for hackathon evaluation purposes.


