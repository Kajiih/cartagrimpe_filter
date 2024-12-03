"""Main module."""

import os
import re
import time
from datetime import datetime
from enum import StrEnum, auto
from pathlib import Path
from urllib.parse import urljoin

import attrs
import json5
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup, ResultSet, Tag
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from loguru import logger

REQUEST_TIMEOUT = 10
URL = "https://cartagrimpe.fr/en/calendrier"
GEOCODE_CACHE_FILE = Path("geocode_cache.json")


class EventType(StrEnum):
    """Types of events."""

    CONTEST = auto()
    COMPETITION = "compétition"
    FILM = auto()
    GATHERING = "rassemblement"
    OTHER = auto()


@attrs.frozen
class Event:
    """An event."""

    name: str
    url: str
    address: str
    date_start: float
    date_end: float
    type: EventType


def fetch_events(url: str) -> str:
    """Fetch the HTML content from the given URL."""
    response = requests.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.text


def get_next_page_url(soup: BeautifulSoup) -> str | None:
    """Find the relative URL of the next page in the pagination controls."""
    pagination_div = soup.find("div", class_="pagination")
    if not pagination_div:
        return None

    next_page_link_regex = r"Next page|Next|Suivant"
    next_link = pagination_div.find("a", text=re.compile(next_page_link_regex, re.IGNORECASE))  # pyright: ignore[reportCallIssue]

    next_link = next_link.get("href") if next_link else None  # pyright: ignore[reportAttributeAccessIssue]

    return next_link  # pyright: ignore[reportReturnType]


def scrap_events(initial_url: str) -> list[Event]:
    """Scrap all event divs from the paginated HTML content starting from the initial URL."""
    all_events: list[Event] = []
    current_url = initial_url
    page_number = 1

    while current_url:
        logger.info(f"Fetching page {page_number}: {current_url}")
        try:
            response = requests.get(current_url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
        except requests.RequestException:
            logger.exception(f"Failed to fetch page {current_url}")
            break
        html_content = response.text

        soup = BeautifulSoup(html_content, "html.parser")

        event_divs: ResultSet[Tag] = soup.find_all("div", class_="event")

        # Parse events on the current page
        for div in event_divs:
            event = parse_event_div(div)
            all_events.append(event)
            logger.debug(f"Scraped Event: {event.name}")

        logger.info(f"Extracted {len(event_divs)} events from page {page_number}.")

        next_page_url = get_next_page_url(soup)
        current_url = urljoin(current_url, next_page_url) if next_page_url else None
        page_number += 1

    logger.info(f"Total events scraped: {len(all_events)}")
    return all_events


def parse_event_div(event_div: Tag) -> Event:  # noqa: PLR0914
    """Instantiate an event from an event div from the html page."""
    # Name
    name_tag = event_div.find("strong")
    name = name_tag.get_text(strip=True) if name_tag else "N/A"

    # URL
    url_tag = event_div.find("a", href=True)
    url = url_tag["href"] if url_tag else "N/A"  # pyright: ignore[reportArgumentType]

    # Extract address
    address_text = event_div.find(text=re.compile(r"Address:", re.IGNORECASE))
    address = address_text.replace("Address:", "").strip() if address_text else "N/A"  # pyright: ignore[reportOptionalCall, reportAttributeAccessIssue]

    # Extract event type
    type_text = event_div.find(text=re.compile(r"Type:", re.IGNORECASE))
    event_type_str = type_text.replace("Type:", "").strip().lower() if type_text else "other"  # pyright: ignore[reportOptionalCall, reportAttributeAccessIssue]
    event_type = EventType(event_type_str)

    # Extract date string
    date_text = event_div.find(text=re.compile(r"From", re.IGNORECASE))
    # Example format: "From 14/01/2024 to 14/01/2024"
    date_parts = date_text.replace("From", "").split("to") if date_text else ""  # pyright: ignore[reportOptionalCall, reportAttributeAccessIssue]
    start_date_str = date_parts[0].strip()
    end_date_str = date_parts[1].strip() if len(date_parts) > 1 else start_date_str

    # Define the date format based on the example
    date_format = "%d/%m/%Y"

    # Parse start and end dates into datetime objects
    date_start_ts = datetime.strptime(start_date_str, date_format).timestamp()
    date_end_ts = datetime.strptime(end_date_str, date_format).timestamp()

    return Event(
        name=name,
        url=url,  # pyright: ignore[reportArgumentType]
        address=address,
        date_start=date_start_ts,
        date_end=date_end_ts,
        type=event_type,
    )


def get_current_location() -> tuple[float, float] | None:
    """Get the current location based on the IP address using ip-api.com."""
    try:
        response = requests.get("http://ip-api.com/json/", timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException:
        logger.exception(f"Error fetching IP Geolocation")
        return None

    data = response.json()
    latitude = data["lat"]
    longitude = data["lon"]
    logger.info(f"Detected current location: ({latitude}, {longitude})")
    return (latitude, longitude)


def geocode_address(address: str, geolocator: Nominatim, delay: float = 0.1):
    """Geocode the given address and return (latitude, longitude)."""
    try:
        location = geolocator.geocode(address)
    except Exception as e:
        logger.error(f"Exception during geocoding address '{address}': {e}")
        return None
    else:
        if location:
            logger.info(f"Geocoded '{address}' to ({location.latitude}, {location.longitude})")
            return (location.latitude, location.longitude)
        logger.warning(f"Geocoding failed for address: {address}")
        return None
    finally:
        time.sleep(delay)  # Respect rate limits


def load_geocode_cache() -> dict:
    """
    Load the geocode cache from a JSON file.

    Returns:
        A dictionary mapping addresses to their (latitude, longitude).
    """
    if not GEOCODE_CACHE_FILE.exists():
        return {}

    with GEOCODE_CACHE_FILE.open() as f:
        return json5.load(f)


def save_geocode_cache(cache: dict) -> None:
    """
    Save the geocode cache to a JSON file.

    Args:
        cache: The cache dictionary to save.
    """
    with GEOCODE_CACHE_FILE.open("w") as f:
        json5.dump(cache, f, ensure_ascii=False, indent=4)
    logger.info(f"Geocode cache saved to {GEOCODE_CACHE_FILE}.")


def add_geocodes(df: pd.DataFrame, geolocator: Nominatim, cache: dict) -> pd.DataFrame:
    """
    Add latitude and longitude columns to the DataFrame by geocoding addresses.

    Args:
        df: The DataFrame containing event addresses.
        geolocator: The geolocator instance.
        cache: The geocode cache.

    Returns:
        The DataFrame with added 'latitude' and 'longitude' columns.
    """
    latitudes = []
    longitudes = []
    for address in df["address"]:
        if address != "N/A":
            # Check cache first
            if hit := cache.get(address):
                lat, lon = hit
                latitudes.append(lat)
                longitudes.append(lon)
                logger.info(f"Address '{address}' found in cache.")
            else:
                # Geocode
                coords = geocode_address(address, geolocator)
                if coords:
                    lat, lon = coords
                    latitudes.append(lat)
                    longitudes.append(lon)
                    cache[address] = coords  # Update cache
                else:
                    latitudes.append(None)
                    longitudes.append(None)
        else:
            latitudes.append(None)
            longitudes.append(None)
    df["latitude"] = latitudes
    df["longitude"] = longitudes
    return df


def compute_distances(df: pd.DataFrame, base_coords: tuple) -> pd.DataFrame:
    """Compute distances from base_coords to each event and add as a new column."""
    distances = []
    for _, row in df.iterrows():
        event_coords = (row["latitude"], row["longitude"])
        if not any(np.isnan(event_coords)):
            distance = geodesic(base_coords, event_coords).kilometers
            distances.append(distance)
        else:
            distances.append(None)
    df["distance_km"] = distances
    return df


def main() -> None:
    """Scrap events from Cartagrimpe."""
    # Scrap events
    events = scrap_events(URL)
    # events = events[:10]  # Temp: Test
    df_events = pd.DataFrame([attrs.asdict(event) for event in events])
    logger.info(f"Parsed {len(df_events)} events.")

    # Initialize geolocator
    geolocator = Nominatim(user_agent="climbing_events_scraper")

    # Geocode event addresses
    logger.info("Starting geocoding of event addresses...")
    geocode_cache = load_geocode_cache()
    df_events = add_geocodes(df_events, geolocator, geocode_cache)

    # Get current location
    current_location = get_current_location()
    if current_location:
        # Format the default address
        default_address = f"{current_location[0]}, {current_location[1]}"
        user_input = input(
            f"Detected current location: {default_address}\nPress Enter to use this location or enter a different address: "
        ).strip()
        if user_input:
            base_address = user_input
            base_location = geocode_address(base_address, geolocator)
            if not base_location:
                logger.error(f"Failed to geocode base location: {base_address}")
                print(f"Failed to geocode base location: {base_address}")
                return
            base_coords = base_location
        else:
            # Use detected coordinates as base_coords
            base_coords = current_location
    else:
        # If detection failed, prompt the user to enter the address
        base_address = input("Enter your base location address (e.g., 'Paris, France'): ").strip()
        base_location = geocode_address(base_address, geolocator)
        if not base_location:
            logger.error(f"Failed to geocode base location: {base_address}")
            print(f"Failed to geocode base location: {base_address}")
            return
        base_coords = base_location  # Tuple (latitude, longitude)

    # Compute distances
    logger.info("Computing distances from base location to events...")
    df_events = compute_distances(df_events, base_coords)
    save_geocode_cache(geocode_cache)

    # Filter by distance
    max_distance = float(input("Enter the maximum distance (km) to filter events: "))

    # Apply filtering
    df_filtered = df_events[df_events["distance_km"] <= max_distance].copy()
    logger.info(f"Filtered down to {len(df_filtered)} events within {max_distance} km.")

    if df_filtered.empty:
        print("No events found within the specified distance.")
        return

    # Choose export format
    export_format = input("Choose export format (csv/excel): ").strip().lower()
    if export_format == "csv":
        output_file = "climbing_events.csv"
        df_filtered.to_csv(output_file, index=False)
    elif export_format == "excel":
        output_file = "climbing_events.xlsx"
        df_filtered.to_excel(output_file, index=False)
    else:
        print("Unsupported format selected.")
        return

    logger.info(f"Exported filtered events to {output_file}.")
    print(f"Exported filtered events to {output_file}.")


if __name__ == "__main__":
    main()
