"""Main module."""

from __future__ import annotations

import re
import sys
import time
from datetime import datetime, timedelta
from enum import StrEnum, auto
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, cast, reveal_type
from urllib.parse import urljoin

import attrs
import json5
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup, ResultSet, Tag
from geopy.geocoders import Nominatim
from loguru import logger
from requests import HTTPError, RequestException

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from geopy import Location

REQUEST_TIMEOUT = 10
EVENT_CALENDAR_URL = "https://cartagrimpe.fr/en/calendrier"
GEOCODE_CACHE_FILE = Path("geocode_cache.json5")


class EventType(StrEnum):
    """Types of events."""

    CONTEST = auto()
    COMPETITION = "compétition"
    FILM = auto()
    GATHERING = "rassemblement"
    OTHER = "autre"


# @attrs.frozen
@attrs.define
class Event:
    """An event."""

    name: str = attrs.field()
    url: str
    address: str
    date_start: str
    date_end: str
    timestamp_start: float
    timestamp_end: float
    type: EventType

    @classmethod
    def from_event_div(cls, tag: Tag) -> Event:  # noqa: PLR0914
        """Initialize the event from an event div from the html page."""
        # Name
        name_tag = tag.find("strong")
        name = name_tag.get_text(strip=True) if name_tag else "N/A"

        # URL
        url_tag = tag.find("a", href=True)
        url = url_tag["href"] if url_tag else "N/A"  # pyright: ignore[reportArgumentType]

        # Extract address
        address_text = tag.find(string=re.compile(r"Address:", re.IGNORECASE))
        address = address_text.replace("Address:", "").strip() if address_text else "N/A"  # pyright: ignore[reportOptionalCall, reportAttributeAccessIssue]

        # Extract event type
        type_text = tag.find(string=re.compile(r"Type:", re.IGNORECASE))
        event_type_str = type_text.replace("Type:", "").strip().lower() if type_text else "other"  # pyright: ignore[reportOptionalCall, reportAttributeAccessIssue]
        event_type = EventType(event_type_str)

        # Extract date string
        date_text = tag.find(string=re.compile(r"From", re.IGNORECASE))
        # Example format: "From 14/01/2024 to 14/01/2024"
        date_parts = date_text.replace("From", "").split("to") if date_text else ""  # pyright: ignore[reportOptionalCall, reportAttributeAccessIssue]
        date_start = date_parts[0].strip()
        date_end = date_parts[1].strip() if len(date_parts) > 1 else date_start

        date_format = "%d/%m/%Y"
        timestamp_start = datetime.strptime(date_start, date_format).timestamp()
        timestamp_end = datetime.strptime(date_end, date_format).timestamp()

        return cls(
            name=name,
            url=url,  # pyright: ignore[reportArgumentType]
            address=address,
            date_start=date_start,
            date_end=date_end,
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            type=event_type,
        )


BOLD = "\033[1m"
CYAN = "\033[36m"
RESET = "\033[0m"

type Coordinate = tuple[float, float]


def get_next_page_url(soup: BeautifulSoup) -> str | None:
    """Find the relative URL of the next page in the pagination controls."""
    pagination_div = soup.find("div", class_="pagination")
    if not pagination_div:
        return None

    next_page_link_regex = r"Next page|Next|Suivant"
    next_link = pagination_div.find("a", string=re.compile(next_page_link_regex, re.IGNORECASE))  # pyright: ignore[reportCallIssue]

    next_link = next_link.get("href") if next_link else None  # pyright: ignore[reportAttributeAccessIssue]

    return next_link  # pyright: ignore[reportReturnType]


def scrap_events(
    initial_url: str, start_date: str = "", nb_pages: int | None = None
) -> list[Event]:
    """Scrap all event divs from the paginated HTML content starting from the initial URL."""
    all_events: list[Event] = []
    current_url = urljoin(initial_url, f"?page=1&date_debut={start_date}")

    page_number = 1

    while current_url and (nb_pages is None or nb_pages - page_number >= 0):
        logger.debug(f"Fetching page {page_number}: {current_url}")
        try:
            response = requests.get(current_url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
        except (RequestException, HTTPError):
            logger.exception(f"Failed to fetch page {current_url}")
            break
        html_content = response.text

        soup = BeautifulSoup(html_content, "html.parser")

        event_divs: ResultSet[Tag] = soup.find_all("div", class_="event")

        # Parse events on the current page
        for div in event_divs:
            event = Event.from_event_div(div)
            all_events.append(event)
            logger.debug(f"Scraped Event: {event.name}")

        logger.debug(f"Extracted {len(event_divs)} events from page {page_number}.")

        next_page_url = get_next_page_url(soup)
        current_url = (
            urljoin(current_url, next_page_url + f"&date_debut={start_date}")
            if next_page_url
            else None
        )
        page_number += 1

    return all_events


def get_current_location() -> Coordinate | None:
    """Get the current location based on the IP address using ip-api.com."""
    try:
        response = requests.get("http://ip-api.com/json/", timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except (RequestException, HTTPError):
        logger.exception(f"Error fetching IP Geolocation")
        return None

    data = response.json()
    latitude = data["lat"]
    longitude = data["lon"]
    logger.debug(f"Detected current location: ({latitude}, {longitude})")

    return (latitude, longitude)


def normalize_address(address: str) -> str:
    """
    Normalize the address by removing unnecessary components and standardizing formatting.

    Args:
        address: The raw address string.

    Returns:
        The normalized address string.
    """
    # Remove "Niveau" and any following number
    address = re.sub(r"Niveau\s*\d+", "", address, flags=re.IGNORECASE)

    # Replace Cr by Cours
    address = re.sub(r"\bCr\b", "Cours", address)

    # Trim whitespace
    address = address.strip()

    return address


def get_city[T](location: Location, default: T = None) -> str | T:
    """Return the city or state/country if city is unavailable."""
    raw_address = location.raw["address"]
    keys = ["city", "town", "municipality", "village", "state", "country"]

    # Return the first non-None value from raw_address
    return next((raw_address.get(key) for key in keys if raw_address.get(key) is not None), default)


def get_amenity[T](location: Location, default: T = None) -> str | T:
    """Return the amenity or house number if city is unavailable."""
    raw_address = location.raw["address"]
    keys = ["amenity", "house_number"]

    # Return the first non-None value from raw_address
    return next((raw_address.get(key) for key in keys if raw_address.get(key) is not None), default)


def find_address(coordinate: Coordinate, geolocator: Nominatim) -> str:
    """Return the address string given the coordinate."""
    location: Location = geolocator.reverse(coordinate)  # pyright: ignore[reportAssignmentType]

    def format_address(location: Location) -> str:
        address = location.raw["address"]
        return f"{get_amenity(location)}, {address['road']}, {get_city(location)}, {address['country']}"

    return format_address(location)


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
        json5.dump(cache, f, indent=4, ensure_ascii=False)
    logger.debug(f"Geocode cache saved to {GEOCODE_CACHE_FILE}.")


class CoordTown(NamedTuple):
    """Tuple containing coordinate and the town they represent."""

    coords: Coordinate
    city: str


# TODO? Add failed geocoding to cache ?
def geocode_address(
    address: str, geolocator: Nominatim, cache: dict[str, CoordTown]
) -> CoordTown | None:
    """Geocode the given address and return the coordinate with a persistent cache."""
    address = normalize_address(address)
    # Check cache
    cached_address = address.lower()
    if hit := cache.get(cached_address):
        logger.debug(f"Address '{address}' found in cache.")
        return CoordTown(hit[0], hit[1])

    # Geocode
    logger.warning(f"Address '{address}' not found in cache.")
    try:
        location: Location = geolocator.geocode(address, addressdetails=True)  # pyright: ignore[reportAssignmentType]
    except Exception:
        logger.exception(f"Exception during geocoding address '{address}'")
        return None

    if not location:
        logger.warning(f"Geocoding failed for address: {address}")
        return None

    coords = (location.latitude, location.longitude)

    city = get_city(location)
    if city is None:
        print("NO")
    assert city is not None

    logger.debug(f"Geocoded '{address}' to {coords}")
    cache[cached_address] = CoordTown(coords, city)

    return CoordTown((location.latitude, location.longitude), city)


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
    cities = []
    for address in df["address"]:
        if address != "N/A" and (location := geocode_address(address, geolocator, cache)):
            coords = location.coords
            latitudes.append(coords[0])
            longitudes.append(coords[1])
            cities.append(location.city)
        else:
            latitudes.append(None)
            longitudes.append(None)
            cities.append(None)
    df["latitude"] = latitudes
    df["longitude"] = longitudes
    df["Ville"] = cities

    return df


class TableMatrices(NamedTuple):
    """Tuple of distance and duration matrices."""

    distance: np.ndarray
    duration: np.ndarray


def get_table_matrices(  # noqa: PLR0914
    sources: list[Coordinate], destinations: list[Coordinate]
) -> TableMatrices:
    """Request distance and duration matrix from the OSRM API."""
    OSRM_API_URL = "http://router.project-osrm.org/table/v1/driving"
    BATCH_SIZE = 250

    def batch[T: Sequence](seq: T, n: int) -> Iterator[T]:
        l = len(seq)
        for ndx in range(0, l, n):
            yield cast("T", seq[ndx : min(ndx + n, l)])

    distance_matrix = np.full((len(sources), len(destinations)), None)
    duration_matrix = np.full((len(sources), len(destinations)), None)

    i = 0
    for source_batch in batch(sources, BATCH_SIZE):
        nb_sources = len(source_batch)
        j = 0
        for destination_batch in batch(destinations, BATCH_SIZE - nb_sources + 1):
            nb_destinations = len(destination_batch)
            # Format coordinates for OSRM API
            source_indexes = ";".join(map(str, range(nb_sources)))
            destination_indexes = ";".join(
                map(str, range(nb_sources, nb_destinations + nb_sources))
            )
            all_coords = source_batch + destination_batch
            coords_str = ";".join([f"{lon},{lat}" for lat, lon in all_coords])

            # Requesting distance and duration matrix
            url = f"{OSRM_API_URL}/{coords_str}?sources={source_indexes}&destinations={destination_indexes}&annotations=duration,distance"
            # url += "&fallback_speed=10"

            logger.info(
                f"Computing distances and durations for {nb_sources} sources and {nb_destinations} destinations."
            )
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()

            # Fill the matrices
            matrix_slice = (slice(i, i + nb_sources), slice(j, j + nb_destinations))
            distance_matrix[matrix_slice] = np.asarray(data["distances"])
            duration_matrix[matrix_slice] = np.asarray(data["durations"])

            j += nb_destinations
        i += nb_sources

    return TableMatrices(distance_matrix, duration_matrix)


def compute_distances(df: pd.DataFrame, source_coords: Coordinate) -> pd.DataFrame:
    """
    Compute distances and durations from the source coordinate to each event using OSRM API.

    Args:
        df: The DataFrame containing event coordinates.
        source_coords: The coordinates of the source.

    Returns:
        The DataFrame with added 'distance_km' and 'duration_sec' columns.
    """
    # Collect coordinates to be sent to OSRM API
    destination_coords = list(zip(df["latitude"], df["longitude"], strict=False))

    table_matrices = get_table_matrices([source_coords], destination_coords)

    # Extract durations and distances from OSRM response
    durations: np.ndarray = table_matrices.duration[0]
    distances: np.ndarray = table_matrices.distance[0]

    def format_duration(s: int) -> str:
        hours, remainder = divmod(s, 3600)
        minutes, _ = divmod(remainder, 60)
        return f"{hours}h {minutes:02d}m"

    # df["Trajet"] = str(timedelta(seconds=durations))
    df["Trajet"] = [
        format_duration(int(duration)) if duration is not None else None for duration in durations
    ]
    df["Distance"] = [
        distance / 1000 if distance is not None else None for distance in distances
    ]  # Convert meters to kilometers

    return df


@logger.catch(reraise=True)
def main() -> None:  # noqa: PLR0915
    """Scrap and filter events from Cartagrimpe."""
    logger.remove()
    logger.add(
        "logs/app.log",
        level="TRACE",
        rotation="1 week",
        compression="zip",
    )

    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{message}</level>",
    )

    # Scrap events
    logger.info("Scrapping events...")
    start_date = datetime.today().strftime("%Y-%m-%d")
    events = scrap_events(EVENT_CALENDAR_URL, start_date=start_date, nb_pages=None)
    logger.info(f"Found {len(events)} events")

    # Filter by type
    filtered_out_types = {EventType.FILM, EventType.GATHERING}
    events = [event for event in events if event.type not in filtered_out_types]
    logger.info(f"{len(events)} events remaining after filtering out {filtered_out_types}")

    df_events = pd.DataFrame([attrs.asdict(event) for event in events])

    # Initialize geolocator
    geolocator = Nominatim(user_agent="climbing_events_scraper")

    # Geocode event addresses
    logger.debug("Starting geocoding of event addresses...")
    geocode_cache = load_geocode_cache()
    df_events = add_geocodes(df_events, geolocator, geocode_cache)

    # Get current location
    current_location = get_current_location()
    if current_location:
        # Format the default address
        default_address = find_address(current_location, geolocator)
        user_input = input(
            f"\nDetected current location: {CYAN}{default_address}{RESET}\n"
            "Press Enter to use this location or enter a different address: "
        ).strip()
        if user_input:
            base_address = user_input
            base_location = geocode_address(base_address, geolocator, geocode_cache)
            if not base_location:
                logger.error(f"Failed to geocode base location: {base_address}")
                print(f"Failed to geocode base location: {base_address}")
                return
            base_coords = base_location.coords
        else:
            # Use detected coordinates as base_coords
            base_coords = current_location
    else:
        # If detection failed, prompt the user to enter the address
        base_address = input("Enter your base location address (e.g., 'Paris, France'): ").strip()
        base_location = geocode_address(base_address, geolocator, geocode_cache)
        # TODO: Print Detailed address found
        if not base_location:
            logger.error(f"Failed to geocode base location: {base_address}")
            print(f"Failed to geocode base location: {base_address}")
            return
        base_coords = base_location.coords

    save_geocode_cache(geocode_cache)

    # Compute distances
    logger.debug("Computing distances from base location to events...")
    df_events = compute_distances(df_events, base_coords)

    # Export full data
    df_events.to_csv("climbing_events_full.csv", index=False)

    # Filter by distance
    # max_distance = float(input("Enter the maximum Distance to filter events: ") or 200)
    max_distance = 200

    df_filtered = df_events[df_events["Distance"] <= max_distance].copy()
    logger.info(f"Filtered down to {len(df_filtered)} events within {max_distance} km.")

    if df_filtered.empty:
        print("No events found within the specified distance.")
        return

    df_filtered = df_events

    # Export
    output_file = "climbing_events.csv"
    df_filtered = df_filtered.rename(
        columns={
            Event.date_start.__name__: "Date",  # pyright:ignore[reportAttributeAccessIssue]
            Event.name.__name__: "Nom",  # pyright:ignore[reportAttributeAccessIssue]
            Event.url.__name__: "Lien",  # pyright:ignore[reportAttributeAccessIssue]
            Event.type.__name__: "Type d'évènement",
            Event.address.__name__: "Adresse",  # pyright:ignore[reportAttributeAccessIssue]
        }
    )
    # Add hyperlink
    df_filtered["Nom"] = df_filtered.apply(
        lambda row: f'=HYPERLINK("{row["Lien"]}"; "{row["Nom"]}")', axis=1
    )
    df_filtered["Distance"] = df_filtered["Distance"].map(
        lambda x: f"{int(x)}" if (x is not None and not np.isnan(x)) else ""
    )

    df_filtered = df_filtered[["Date", "Nom", "Ville", "Adresse", "Distance", "Trajet"]]
    df_filtered.sort_values("Trajet", inplace=True)
    df_filtered.to_csv(output_file, index=False)

    logger.info(f"Exported filtered events to {output_file}.")
    print(f"Exported filtered events to {output_file}.")


if __name__ == "__main__":
    main()
