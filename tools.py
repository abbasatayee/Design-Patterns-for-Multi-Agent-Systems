import json
import random
import secrets
import string
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from langchain.tools import tool

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "slight snow",
    73: "moderate snow",
    75: "heavy snow",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    95: "thunderstorm",
}


def _fetch_json(url: str, headers: dict | None = None) -> dict | list:
    request = Request(url, headers=headers or {})
    with urlopen(request, timeout=10) as response:
        return json.load(response)


def _fetch_text(url: str, headers: dict | None = None) -> str:
    request = Request(url, headers=headers or {})
    with urlopen(request, timeout=10) as response:
        return response.read().decode()


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city (temperature, humidity, wind, conditions)."""
    try:
        geo = _fetch_json(f"{GEOCODING_URL}?name={quote(city)}&count=1")
    except (HTTPError, URLError, TimeoutError) as e:
        return f"Could not look up '{city}': {e}"

    results = geo.get("results") or []
    if not results:
        return f"No location found for '{city}'."

    place = results[0]
    name = place.get("name", city)
    country = place.get("country", "")
    lat, lon = place["latitude"], place["longitude"]

    try:
        forecast = _fetch_json(
            f"{FORECAST_URL}?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code"
            "&wind_speed_unit=mph&temperature_unit=fahrenheit"
        )
    except (HTTPError, URLError, TimeoutError) as e:
        return f"Could not fetch weather for {name}: {e}"

    current = forecast["current"]
    code = int(current["weather_code"])
    conditions = WEATHER_CODES.get(code, f"code {code}")
    location = f"{name}, {country}" if country else name
    return (
        f"{location}: {conditions}, "
        f"{current['temperature_2m']}°F, "
        f"humidity {current['relative_humidity_2m']}%, "
        f"wind {current['wind_speed_10m']} mph"
    )


@tool
def wikipedia_summary(topic: str) -> str:
    """Get a short Wikipedia summary for a person, place, concept, or event."""
    try:
        data = _fetch_json(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(topic.replace(' ', '_'))}"
        )
    except HTTPError:
        return f"No Wikipedia article found for '{topic}'."
    except (URLError, TimeoutError) as e:
        return f"Wikipedia lookup failed: {e}"

    title = data.get("title", topic)
    extract = data.get("extract", "No summary available.")
    return f"{title}: {extract}"


@tool
def get_crypto_price(coin: str, currency: str = "usd") -> str:
    """Get the live price of a cryptocurrency (e.g. bitcoin, ethereum, dogecoin)."""
    coin_id = coin.lower().strip().replace(" ", "-")
    currency = currency.lower().strip()
    try:
        data = _fetch_json(
            f"https://api.coingecko.com/api/v3/simple/price"
            f"?ids={quote(coin_id)}&vs_currencies={quote(currency)}"
        )
    except (HTTPError, URLError, TimeoutError) as e:
        return f"Price lookup failed: {e}"

    if coin_id not in data:
        return f"Unknown coin '{coin}'. Try ids like bitcoin, ethereum, solana, dogecoin."
    price = data[coin_id][currency]
    return f"{coin_id}: {price:,.2f} {currency.upper()}"


@tool
def define_word(word: str) -> str:
    """Look up definitions, phonetics, and examples for an English word."""
    try:
        entries = _fetch_json(f"https://api.dictionaryapi.dev/api/v2/entries/en/{quote(word)}")
    except HTTPError:
        return f"No definition found for '{word}'."
    except (URLError, TimeoutError) as e:
        return f"Dictionary lookup failed: {e}"

    entry = entries[0]
    phonetic = entry.get("phonetic", "")
    meanings = entry.get("meanings", [])[:2]
    parts = [f"{word} ({phonetic}):" if phonetic else f"{word}:"]
    for meaning in meanings:
        pos = meaning.get("partOfSpeech", "")
        defs = meaning.get("definitions", [])[:2]
        for d in defs:
            line = f"  [{pos}] {d.get('definition', '')}"
            if d.get("example"):
                line += f' — e.g. "{d["example"]}"'
            parts.append(line)
    return "\n".join(parts)


@tool
def where_is_iss() -> str:
    """Get the International Space Station's current latitude, longitude, and map link."""
    try:
        data = _fetch_json("http://api.open-notify.org/iss-now.json")
    except (HTTPError, URLError, TimeoutError) as e:
        return f"ISS tracker unavailable: {e}"

    pos = data["iss_position"]
    lat, lon = pos["latitude"], pos["longitude"]
    maps = f"https://www.google.com/maps?q={lat},{lon}"
    return f"ISS is at {lat}°N, {lon}°E. Map: {maps}"


@tool
def suggest_activity(activity_type: str = "") -> str:
    """Suggest a random activity when someone is bored. Optional type: outdoor, social, creative, relaxing."""
    url = "https://www.boredapi.com/api/activity"
    if activity_type:
        url += f"?type={quote(activity_type.lower())}"
    try:
        data = _fetch_json(url)
    except (HTTPError, URLError, TimeoutError) as e:
        return f"Activity API unavailable: {e}"

    return (
        f"Try: {data['activity']} "
        f"(type: {data.get('type', 'n/a')}, "
        f"~{data.get('duration', '?')} min, "
        f"participants: {data.get('participants', '?')})"
    )


@tool
def country_facts(country: str) -> str:
    """Get capital, population, region, languages, and fun facts about a country."""
    try:
        data = _fetch_json(f"https://restcountries.com/v3.1/name/{quote(country)}?fields=name,capital,population,region,subregion,languages,currencies,flags")
    except HTTPError:
        return f"Country '{country}' not found."
    except (URLError, TimeoutError) as e:
        return f"Country lookup failed: {e}"

    c = data[0]
    name = c["name"]["common"]
    capital = ", ".join(c.get("capital", ["N/A"]))
    pop = f"{c.get('population', 0):,}"
    langs = ", ".join(c.get("languages", {}).values())
    currencies = ", ".join(
        f"{v['name']} ({k})" for k, v in c.get("currencies", {}).items()
    )
    return (
        f"{name}: capital {capital}, population {pop}, "
        f"{c.get('subregion', c.get('region', ''))}. "
        f"Languages: {langs or 'n/a'}. "
        f"Currencies: {currencies or 'n/a'}."
    )


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert money between currencies using live ECB rates (e.g. USD to EUR)."""
    try:
        data = _fetch_json(
            f"https://api.frankfurter.app/latest"
            f"?amount={amount}&from={quote(from_currency.upper())}&to={quote(to_currency.upper())}"
        )
    except (HTTPError, URLError, TimeoutError) as e:
        return f"Conversion failed: {e}"

    result = data["rates"][to_currency.upper()]
    return f"{amount} {from_currency.upper()} = {result} {to_currency.upper()} (rate date: {data['date']})"


@tool
def world_clock(timezone: str) -> str:
    """Get the current date and time for an IANA timezone (e.g. America/New_York, Europe/Paris, Asia/Tokyo)."""
    tz = timezone.replace(" ", "_")
    try:
        data = _fetch_json(f"https://worldtimeapi.org/api/timezone/{quote(tz)}")
    except HTTPError:
        return f"Unknown timezone '{timezone}'. Use IANA names like America/Los_Angeles."
    except (URLError, TimeoutError) as e:
        return f"Time lookup failed: {e}"

    return f"{data['timezone']}: {data['datetime']} ({data.get('day_of_week', '')})"


@tool
def random_useless_fact() -> str:
    """Return a random interesting (often useless) fact."""
    try:
        data = _fetch_json("https://uselessfacts.jsph.pl/random.json?language=en")
    except (URLError, TimeoutError) as e:
        return f"Fact API unavailable: {e}"
    return data.get("text", "No fact returned.")


@tool
def dad_joke() -> str:
    """Tell a random dad joke."""
    try:
        return _fetch_text(
            "https://icanhazdadjoke.com/",
            headers={"Accept": "text/plain", "User-Agent": "langchain-llm-app"},
        )
    except (HTTPError, URLError, TimeoutError) as e:
        return f"Joke API unavailable: {e}"


@tool
def nasa_picture_of_day() -> str:
    """Get NASA's Astronomy Picture of the Day title, explanation, and image URL."""
    try:
        data = _fetch_json("https://api.nasa.gov/planetary/apod?api_key=DEMO_KEY")
    except (HTTPError, URLError, TimeoutError) as e:
        return f"NASA APOD unavailable: {e}"

    explanation = data.get("explanation", "")[:400]
    if len(data.get("explanation", "")) > 400:
        explanation += "..."
    return (
        f"{data.get('title', 'APOD')} ({data.get('date', '')})\n"
        f"{explanation}\n"
        f"Image: {data.get('url', data.get('hdurl', 'n/a'))}"
    )


@tool
def roll_dice(notation: str = "1d6") -> str:
    """Roll dice using tabletop notation like 2d6, 1d20, or 3d8+2."""
    notation = notation.lower().replace(" ", "")
    bonus = 0
    if "+" in notation:
        notation, bonus_str = notation.split("+", 1)
        bonus = int(bonus_str)
    elif "-" in notation:
        notation, bonus_str = notation.split("-", 1)
        bonus = -int(bonus_str)

    try:
        count_str, sides_str = notation.split("d")
        count = int(count_str) if count_str else 1
        sides = int(sides_str)
    except ValueError:
        return "Use notation like 2d6 or 1d20."

    if count < 1 or count > 20 or sides < 2 or sides > 100:
        return "Roll between 1-20 dice with 2-100 sides."

    rolls = [random.randint(1, sides) for _ in range(count)]
    total = sum(rolls) + bonus
    rolls_str = " + ".join(map(str, rolls))
    bonus_str = f" {'+' if bonus >= 0 else ''}{bonus}" if bonus else ""
    return f"Rolled {notation}: [{rolls_str}]{bonus_str} = {total}"


@tool
def generate_password(length: int = 16, include_symbols: bool = True) -> str:
    """Generate a secure random password. Length 8-64."""
    length = max(8, min(64, int(length)))
    alphabet = string.ascii_letters + string.digits
    if include_symbols:
        alphabet += "!@#$%^&*-_=+"
    password = "".join(secrets.choice(alphabet) for _ in range(length))
    return f"Generated {length}-char password: {password}"


@tool
def days_until(target_date: str) -> str:
    """Count days until a date. Format: YYYY-MM-DD (e.g. 2026-12-25)."""
    try:
        target = datetime.strptime(target_date.strip(), "%Y-%m-%d").date()
    except ValueError:
        return "Use date format YYYY-MM-DD."

    today = datetime.now().date()
    delta = (target - today).days
    if delta > 0:
        return f"{delta} days until {target_date}."
    if delta < 0:
        return f"{target_date} was {-delta} days ago."
    return f"Today is {target_date}!"


GEO_TOOLS = [get_weather, world_clock, where_is_iss, country_facts]
KNOWLEDGE_TOOLS = [wikipedia_summary, define_word, random_useless_fact]
FINANCE_TOOLS = [get_crypto_price, convert_currency]
FUN_TOOLS = [
    suggest_activity,
    dad_joke,
    nasa_picture_of_day,
    roll_dice,
    generate_password,
    days_until,
]

ALL_TOOLS = GEO_TOOLS + KNOWLEDGE_TOOLS + FINANCE_TOOLS + FUN_TOOLS
