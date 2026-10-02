"""Agent tools. Each one is a plain Python function plus a JSON schema.
The same schema works for Claude (input_schema) and Ollama/OpenAI (parameters)."""
import json
import os
from datetime import datetime
from pathlib import Path

import httpx

NOTES = Path(os.getenv("ASSIST_NOTES", "~/assistant/notes.md")).expanduser()
HA_URL = os.getenv("HA_URL", "http://homeassistant.local:8123")
HA_TOKEN = os.getenv("HA_TOKEN", "")


def weather(city: str) -> dict:
    """Current conditions and today's forecast via Open-Meteo (no API key)."""
    geo = httpx.get("https://geocoding-api.open-meteo.com/v1/search",
                    params={"name": city, "count": 1, "language": "en"}, timeout=10).json()
    if not geo.get("results"):
        return {"error": f"city '{city}' not found"}
    g = geo["results"][0]
    w = httpx.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": g["latitude"], "longitude": g["longitude"],
        "current": "temperature_2m,precipitation,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "forecast_days": 1, "timezone": "auto"}, timeout=10).json()
    return {"city": g["name"], "now": w["current"], "today": w["daily"]}


def take_note(text: str) -> dict:
    NOTES.parent.mkdir(parents=True, exist_ok=True)
    with NOTES.open("a", encoding="utf-8") as f:
        f.write(f"- {datetime.now():%Y-%m-%d %H:%M} {text}\n")
    return {"ok": True}


def read_notes(last: int = 10) -> dict:
    if not NOTES.exists():
        return {"notes": []}
    return {"notes": NOTES.read_text(encoding="utf-8").splitlines()[-last:]}


def home(domain: str, service: str, entity_id: str) -> dict:
    """Call a Home Assistant service, e.g. light / turn_on / light.living_room."""
    if domain not in {"light", "switch", "cover", "media_player", "scene"}:
        return {"error": "domain not allowed"}  # allowlist: the model does not get to pick anything it likes
    r = httpx.post(f"{HA_URL}/api/services/{domain}/{service}",
                   headers={"Authorization": f"Bearer {HA_TOKEN}"},
                   json={"entity_id": entity_id}, timeout=10)
    return {"status": r.status_code}


FUNCS = {"weather": weather, "take_note": take_note, "read_notes": read_notes, "home": home}

SCHEMAS = [
    {"name": "weather", "description": "Current weather and today's forecast for a city.",
     "schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}},
    {"name": "take_note", "description": "Save a note or reminder for the user.",
     "schema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
    {"name": "read_notes", "description": "Read the most recent saved notes.",
     "schema": {"type": "object", "properties": {"last": {"type": "integer"}}}},
    {"name": "home", "description": "Control Home Assistant devices (lights, plugs, blinds, media, scenes).",
     "schema": {"type": "object", "properties": {
         "domain": {"type": "string"}, "service": {"type": "string"}, "entity_id": {"type": "string"}},
         "required": ["domain", "service", "entity_id"]}},
]


def run_tool(name: str, args: dict) -> str:
    try:
        return json.dumps(FUNCS[name](**args), ensure_ascii=False)
    except Exception as e:  # the error goes back to the model, which decides what to say
        return json.dumps({"error": str(e)}, ensure_ascii=False)
