from dataclasses import dataclass
from typing import Dict, List, Optional

import requests


class EbirdApiError(Exception):
    pass


@dataclass
class EbirdApiSpeciesMatch:
    sci_name: str
    com_name: str
    species_code: str


class EbirdApiClient:
    def __init__(self, api_token: str, timeout_seconds: int = 15) -> None:
        self._api_token = (api_token or "").strip()
        self._timeout_seconds = timeout_seconds
        self._taxonomy_rows: Optional[List[dict]] = None
        self._session = requests.Session()
        self._session.headers.update(
            {
                "X-eBirdApiToken": self._api_token,
                "User-Agent": "bird-identifier/1.0",
            }
        )

    @property
    def enabled(self) -> bool:
        return bool(self._api_token)

    def _load_taxonomy_species(self) -> List[dict]:
        if self._taxonomy_rows is not None:
            return self._taxonomy_rows
        response = self._session.get(
            "https://api.ebird.org/v2/ref/taxonomy/ebird",
            params={"fmt": "json", "cat": "species"},
            timeout=max(self._timeout_seconds, 60),
        )
        if response.status_code in (401, 403):
            raise EbirdApiError("api_auth_failed")
        if response.status_code >= 400:
            raise EbirdApiError(f"taxonomy_http_{response.status_code}")
        try:
            rows = response.json()
        except ValueError as exc:
            raise EbirdApiError("invalid_json") from exc
        if not isinstance(rows, list):
            raise EbirdApiError("taxonomy_not_list")
        self._taxonomy_rows = rows
        return rows

    def find_species(self, candidate_name: str) -> Optional[EbirdApiSpeciesMatch]:
        if not self.enabled:
            return None
        name = (candidate_name or "").strip()
        if not name:
            return None

        # /v2/ref/taxonomy/find is not available on current API; match against full species taxonomy.
        try:
            rows = self._load_taxonomy_species()
        except EbirdApiError:
            raise

        lowered = name.lower()
        exact: Optional[EbirdApiSpeciesMatch] = None
        partial: Optional[EbirdApiSpeciesMatch] = None
        for item in rows:
            sci_name = str(item.get("sciName", "")).strip()
            com_name = str(item.get("comName", "")).strip()
            species_code = str(item.get("speciesCode", "")).strip()
            if not species_code:
                continue
            match = EbirdApiSpeciesMatch(sci_name=sci_name, com_name=com_name, species_code=species_code)
            if sci_name.lower() == lowered or com_name.lower() == lowered:
                exact = match
                break
            if partial is None and (lowered in sci_name.lower() or lowered in com_name.lower()):
                partial = match
        return exact or partial

    def recent_observations(
        self, region_code: str, species_code: str, max_rows: int = 10
    ) -> List[Dict[str, object]]:
        """Return recent observation rows (location, date, count) from eBird API."""
        if not self.enabled:
            return []
        region = (region_code or "").strip().upper()
        species = (species_code or "").strip().lower()
        if not region or not species:
            return []

        response = self._session.get(
            f"https://api.ebird.org/v2/data/obs/{region}/recent/{species}",
            timeout=self._timeout_seconds,
        )
        if response.status_code in (401, 403):
            raise EbirdApiError("api_auth_failed")
        if response.status_code >= 400:
            return []
        try:
            rows = response.json()
        except ValueError:
            return []
        if not isinstance(rows, list):
            return []

        out: List[Dict[str, object]] = []
        cap = max(1, min(max_rows, 100))
        for item in rows[:cap]:
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "obsDt": str(item.get("obsDt", "")).strip(),
                    "locName": str(item.get("locName", "")).strip(),
                    "lat": item.get("lat"),
                    "lng": item.get("lng"),
                    "howMany": item.get("howMany"),
                    "subnational1Code": str(item.get("subnational1Code", "")).strip(),
                }
            )
        return out

    def recent_observation_count(self, region_code: str, species_code: str) -> int:
        return len(self.recent_observations(region_code, species_code, max_rows=500))

    def region_recent_observations(
        self, species_code: str, regions: List[str], max_per_region: int
    ) -> Dict[str, List[Dict[str, object]]]:
        """Recent obs per region code (e.g. HK, CN)."""
        result: Dict[str, List[Dict[str, object]]] = {}
        for region in regions:
            try:
                result[region] = self.recent_observations(region, species_code, max_rows=max_per_region)
            except EbirdApiError:
                result[region] = []
        return result

    def region_evidence(self, species_code: str, regions: List[str]) -> Dict[str, int]:
        evidence: Dict[str, int] = {}
        for region in regions:
            try:
                evidence[region] = len(
                    self.recent_observations(region, species_code, max_rows=500)
                )
            except EbirdApiError:
                evidence[region] = -1
        return evidence

