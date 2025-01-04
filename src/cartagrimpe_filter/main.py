"""Main module."""

from __future__ import annotations

import re
import sys
from datetime import datetime, timedelta
from enum import Enum, StrEnum, auto
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    NamedTuple,
    TypedDict,
    cast,
    override,
)
from urllib.parse import urljoin

import attrs
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup, ResultSet, Tag
from cachier import cachier
from geopy import Location  # pyright:ignore[reportMissingTypeStubs]
from geopy.geocoders import Nominatim  # pyright:ignore[reportMissingTypeStubs]
from kajihs_utils import batch, get_first
from loguru import logger
from requests import HTTPError, RequestException

from cartagrimpe_filter.__about__ import __app_name__
from cartagrimpe_filter.gsheet import update_spreadsheet

if TYPE_CHECKING:
    from numpy.typing import NDArray


# === User Config (temp) ===
BASE_ADDRESS = "MCJ de Migennes"
G_SHEET_KEY = "1BgeKD8rEr9TV1p7V9vni8stgSki1cl_eqcVxebXTfSo"


# === Constants ===
OUTPUT_FILE = Path("output/climbing_events.csv")
REQUEST_TIMEOUT = 10
EVENT_CALENDAR_URL = "https://cartagrimpe.fr/en/calendrier"
_CACHE_STALE_TIME = timedelta(weeks=4)
CLEAR_GEOCODE_CACHES = False


class _Auto(Enum):
    """
    Sentinel to indicate a value automatically set.

    Can be used as type annotation with typing.Literal[AUTO] to show that a
    value may be AUTO.
    """

    AUTO = auto()

    @override
    def __repr__(self) -> Literal["AUTO"]:
        return "AUTO"

    def __bool__(self) -> Literal[False]:
        return False


AUTO = _Auto.AUTO
"""
Sentinel to indicate a value automatically set.
"""
type Coordinate = tuple[float, float]


class TableHeader(StrEnum):
    """Headers of the formatted data table."""

    DATE = "Date (timestamp)"
    NAME = "Nom"
    PLACE = "Lieu"
    ADDRESS = "Adresse"
    DISTANCE = "Distance (m)"
    DURATION = "Trajet (s)"
    TYPE = "Type"


class EventType(StrEnum):
    """Types of events."""

    CONTEST = "contest"
    COMPETITION = "compétition"
    FILM = "film"
    GATHERING = "rassemblement"
    OTHER = "autre"


@attrs.frozen
class IPLocation:
    """Geolocation data obtained from IP Geolocation."""

    lat: float
    long: float
    city: str

    @classmethod
    def from_ipinfo_dict(cls, d: dict[str, Any]) -> IPLocation:
        """Initialize from data obtained form http://ipinfo.io."""
        lat, long = [float(x) for x in d["loc"].split(",")]

        return cls(
            lat=lat,
            long=long,
            city=d["city"],
        )


DEFAULT_IP_LOCATION_DICT = {
    "city": "Migennes",
    "country": "FR",
    "hostname": "_",
    "ip": "_",
    "loc": "47.9655,3.5179",
    "org": "_",
    "postal": "89400",
    "readme": "https://ipinfo.io/missingauth",
    "region": "Bourgogne-Franche-Comté",
    "timezone": "Europe/Paris",
}

DEFAULT_IP_LOCATION = IPLocation.from_ipinfo_dict(DEFAULT_IP_LOCATION_DICT)

DEFAULT_LOCATION = Location(address="Migennes", point=(47.9655, 3.5179), raw={"address": {}})


# TODO: Replace sentinels cached values by None to not cache them
# TODO? replace auto initializations with a single factory method
# TODO? use cachier(pickle_reload=False) if everything runs in a single thread
@attrs.frozen
class Client:
    """A client to store geolocator, base location etc."""

    geolocator: Nominatim
    base_address: str = AUTO
    today_date: str = attrs.field(init=False, factory=lambda: datetime.today().strftime("%Y-%m-%d"))
    ip_location: IPLocation = attrs.field(init=False)
    base_location: Location = attrs.field(init=False)

    @ip_location.default  # pyright:ignore[reportAttributeAccessIssue, reportUntypedFunctionDecorator]
    def _ip_location_factory(self) -> IPLocation:
        return self.get_current_ip_location()

    @base_location.default  # pyright:ignore[reportAttributeAccessIssue, reportUntypedFunctionDecorator]
    def _base_location_factory(self) -> Location:
        if self.base_address == AUTO:
            coords = self.ip_location.lat, self.ip_location.long
            location = self.reverse_geocode(coords)

            # Go around frozen attribute to update base address
            attr_name = Client.base_address.__name__  # pyright: ignore[reportAttributeAccessIssue]
            object.__setattr__(self, attr_name, location.address)  # noqa: PLC2801
            return location

        return self.geocode(self.base_address)

    @classmethod
    def from_app_name(cls, app_name: str) -> Client:
        """Return a client with initialized geolocator."""
        return cls(
            Nominatim(
                user_agent=app_name,
                timeout=REQUEST_TIMEOUT,  # pyright: ignore[reportArgumentType]
            )
        )

    @property
    def base_coords(self) -> Coordinate:
        """Coordinates of the base location."""
        return (self.base_location.latitude, self.base_location.longitude)

    def geocode(self, address: str) -> Location:
        """Return the geocoded location of the given address."""
        return self._geocode_normalized(self.normalize_address(address))

    @cachier(stale_after=_CACHE_STALE_TIME)
    def _geocode_normalized(self, address: str) -> Location:
        logger.debug(f"Address '{address}' not found in cache.")

        try:
            location: Location | None = self.geolocator.geocode(address, addressdetails=True)  # pyright: ignore[reportAssignmentType]
        except Exception:
            logger.exception(
                f"Exception during geocoding address '{address}'. Falling back to default location."
            )
            return DEFAULT_LOCATION

        if location is None:
            logger.warning(
                f"Geocoding returned None for address '{address}'. Falling back to default location."
            )
            return DEFAULT_LOCATION

        return location

    @cachier(stale_after=_CACHE_STALE_TIME)
    def reverse_geocode(self, coords: Coordinate) -> Location:
        """Return the location corresponding to the coordinates."""
        logger.debug(f"Coordinates '{coords}' not found in cache.")

        try:
            location: Location | None = self.geolocator.reverse(coords, addressdetails=True)  # pyright: ignore[reportAssignmentType]
        except Exception:
            logger.exception(
                f"Exception during reverse geocoding address '{coords}'. Falling back to default location."
            )
            return DEFAULT_LOCATION

        if location is None:
            logger.warning(
                f"Reverse geocoding failed for coordinates: {coords}. Falling back to default location."
            )
            return DEFAULT_LOCATION

        return location

    @staticmethod
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

        address = address.strip().lower()

        return address

    @staticmethod
    def get_current_ip_location() -> IPLocation:
        """Get the current location based on the IP address."""
        try:
            response = requests.get("https://ipinfo.io/json", timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
        except (RequestException, HTTPError):
            logger.exception(f"Error fetching IP Geolocation. Falling back to default IP location.")
            return DEFAULT_IP_LOCATION

        ip_location = IPLocation.from_ipinfo_dict(response.json())
        logger.debug(f"Detected current location: ({ip_location.city})")

        return ip_location

    def request_table_lists(self, destinations: list[Coordinate]) -> TableMatrices:
        """Return the distance and duration lists with the client base coordinates as source."""
        table_matrices = request_table_matrices([self.base_coords], destinations)

        return TableMatrices(table_matrices.distance[0], table_matrices.duration[0])


class EventDict(TypedDict):
    """Dictionary of curated and formatted event attributes."""


@attrs.frozen
class Event:
    """An event."""

    client: ClassVar[Client]

    name: str
    url: str
    original_address: str
    timestamp_start: float
    timestamp_end: float
    type: EventType
    location: Location = attrs.field(init=False, repr=False)

    @location.default  # pyright:ignore[reportAttributeAccessIssue,reportUntypedFunctionDecorator]
    def _location_factory(self) -> Location:
        return self.client.geocode(self.original_address)

    @classmethod
    def from_event_div(cls, tag: Tag) -> Event:  # noqa: PLR0914
        """Initialize the event from an event div from the html page."""
        # Name
        name_tag = tag.find("strong")
        name = name_tag.get_text(strip=True) if name_tag else "N/A"

        # URL
        url_tag = tag.find("a", href=True)
        url = cast("str", url_tag["href"]) if url_tag else "N/A"  # pyright: ignore[reportArgumentType]

        # Extract address
        address_text = tag.find(string=re.compile(r"Address:", re.IGNORECASE))
        address = address_text.replace("Address:", "").strip() if address_text else "N/A"  # pyright: ignore[reportOptionalCall, reportAttributeAccessIssue]

        # Extract event type
        type_text = tag.find(string=re.compile(r"Type:", re.IGNORECASE))
        event_type_str = type_text.replace("Type:", "").strip().lower() if type_text else "other"  # pyright: ignore[reportOptionalCall, reportAttributeAccessIssue]
        event_type = EventType(event_type_str)

        # Extract date string
        date_text = tag.find(string=re.compile(r"From", re.IGNORECASE))
        # Example format: "From 14/01/2024 to 15/01/2024"
        date_parts = date_text.replace("From", "").split("to") if date_text else ""  # pyright: ignore[reportOptionalCall, reportAttributeAccessIssue]
        date_start = date_parts[0].strip()
        date_end = date_parts[1].strip() if len(date_parts) > 1 else date_start

        date_format = "%d/%m/%Y"
        timestamp_start = datetime.strptime(date_start, date_format).timestamp()
        timestamp_end = datetime.strptime(date_end, date_format).timestamp()

        return cls(
            name=name,
            url=url,
            original_address=address,
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            type=event_type,
        )

    def as_table_dict(self) -> dict[str, Any]:
        """Return a dict representation formatted with table columns."""
        return {
            TableHeader.DATE: self.timestamp_start,
            TableHeader.NAME: f'=HYPERLINK("{self.url}"; "{self.name}")',
            TableHeader.PLACE: get_city(self.location),
            TableHeader.ADDRESS: get_address_short(self.location),
            TableHeader.TYPE: self.type,
        }


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


def get_city(location: Location) -> str | None:
    """Return the city or state/country if city is unavailable."""
    return get_first(
        location.raw["address"], ["city", "town", "municipality", "village", "state", "country"]
    )


def get_amenity(location: Location) -> str | None:
    """Return the location name or house number."""
    return get_first(location.raw["address"], ["amenity", "house_number"])


def get_address_short(location: Location) -> str:
    """Return a short address."""
    address = location.raw["address"]
    address_split = [get_amenity(location), address.get("road"), get_city(location)]
    address_split = [x for x in address_split if x]

    return ", ".join(address_split) if address_split else address.get("country", "")


class TableMatrices(NamedTuple):
    """
    Tuple of distance and duration matrices.

    Also represents tuple of distance and duration lists.
    """

    distance: NDArray[np.float64]
    duration: NDArray[np.float64]


def request_table_matrices(  # noqa: PLR0914
    sources: list[Coordinate], destinations: list[Coordinate]
) -> TableMatrices:
    """Request distance and duration matrix from the OSRM API."""
    OSRM_API_URL = "http://router.project-osrm.org/table/v1/driving"
    BATCH_SIZE = 250

    distance_matrix = np.full((len(sources), len(destinations)), None)
    duration_matrix = np.full((len(sources), len(destinations)), None)

    i = 0
    for source_batch in batch(sources, BATCH_SIZE - 1):
        nb_sources = len(source_batch)
        j = 0
        for destination_batch in batch(destinations, BATCH_SIZE - nb_sources):
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
            distance_matrix[matrix_slice] = np.asarray(data["distances"], dtype=float)
            duration_matrix[matrix_slice] = np.asarray(data["durations"], dtype=float)

            j += nb_destinations
        i += nb_sources

    return TableMatrices(distance_matrix, duration_matrix)


def setup_logging() -> None:
    """Set up logging in log file and consol."""
    logger.remove()
    logger.add(
        "logs/app.log",
        level="DEBUG",
        rotation="1 week",
        compression="zip",
    )

    logger.add(
        sys.stdout,
        # level="DEBUG",  # Temp: dev
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{message}</level>",
    )


@logger.catch(reraise=True)
def main() -> None:
    """Scrap and filter events from Cartagrimpe."""
    setup_logging()

    # Initialize geolocator client
    Event.client = client = Client.from_app_name(__app_name__)

    if CLEAR_GEOCODE_CACHES:
        client._geocode_normalized.clear_cache()
        client.reverse_geocode.clear_cache()

    # Scrap events
    logger.info("Scrapping and geocoding events...")
    events = scrap_events(EVENT_CALENDAR_URL, start_date=client.today_date, nb_pages=None)
    logger.info(f"Found {len(events)} events")

    events_df = pd.DataFrame(event.as_table_dict() for event in events)

    # Compute  distances and durations from the base location
    destinations = [(event.location.latitude, event.location.longitude) for event in events]
    table_matrices = client.request_table_lists(destinations=destinations)

    events_df[TableHeader.DISTANCE] = [
        int(x) if not np.isnan(x) else x for x in table_matrices.distance
    ]
    events_df[TableHeader.DURATION] = [
        int(x) if not np.isnan(x) else x for x in table_matrices.duration
    ]
    exported_df = events_df.astype(
        {
            TableHeader.DATE: pd.Int32Dtype(),
            TableHeader.DURATION: pd.Int32Dtype(),
            TableHeader.DISTANCE: pd.Int32Dtype(),
        },
        copy=True,
    )

    # Export
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    exported_df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Exported filtered events to {OUTPUT_FILE}.")

    # Update Google Spreadsheet
    update_spreadsheet(events_df, sheet_key=G_SHEET_KEY)
    logger.info(f"Updated spreadsheet at with key: {G_SHEET_KEY}.")


if __name__ == "__main__":
    main()
