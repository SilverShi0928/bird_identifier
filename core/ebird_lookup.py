import re
from dataclasses import dataclass
from html import unescape
from typing import Dict, List, Optional
from urllib.parse import quote

import requests
from core.ebird_api_client import EbirdApiClient, EbirdApiError


class EbirdLookupError(Exception):
    pass


@dataclass
class EbirdLookupResult:
    source_candidate: str
    species_page_url: str
    identification_text: str
    source_mode: str = "web_fallback"
    species_code: str = ""
    region_evidence: Optional[Dict[str, int]] = None


class EbirdLookupService:
    def __init__(self, timeout_seconds: int = 15, api_token: str = "", regions: Optional[List[str]] = None) -> None:
        self._timeout_seconds = timeout_seconds
        self._regions = [x.strip().upper() for x in (regions or ["HK", "CN"]) if x.strip()]
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
                )
            }
        )
        self._api_client = EbirdApiClient(api_token=api_token, timeout_seconds=timeout_seconds)

    def lookup_identification(self, candidate_names: List[str]) -> Optional[EbirdLookupResult]:
        errors: List[str] = []
        for candidate in candidate_names:
            candidate = (candidate or "").strip()
            if not candidate:
                continue
            try:
                result = self._lookup_one(candidate)
                if result:
                    return result
            except EbirdLookupError as exc:
                errors.append(f"{candidate}: {exc}")

        if errors:
            raise EbirdLookupError(" | ".join(errors))
        return None

    def _lookup_one(self, candidate_name: str) -> Optional[EbirdLookupResult]:
        api_species_url = ""
        api_species_code = ""
        api_region_evidence: Dict[str, int] = {}
        source_mode = "web_fallback"
        if self._api_client.enabled:
            try:
                species_match = self._api_client.find_species(candidate_name)
            except EbirdApiError as exc:
                raise EbirdLookupError(str(exc)) from exc
            if species_match:
                api_species_code = species_match.species_code
                api_species_url = f"https://ebird.org/species/{api_species_code}"
                api_region_evidence = self._api_client.region_evidence(api_species_code, self._regions)
                source_mode = "mixed"

        slug = self._slugify(candidate_name)
        species_url = api_species_url or f"https://ebird.org/species/{slug}"
        response = self._session.get(species_url, timeout=self._timeout_seconds, allow_redirects=False)

        # eBird may redirect bots/non-authenticated clients to login.
        if response.status_code in (301, 302, 303, 307, 308):
            location = response.headers.get("Location", "")
            if "login" in location.lower():
                if api_species_code:
                    return EbirdLookupResult(
                        source_candidate=candidate_name,
                        species_page_url=species_url,
                        identification_text=self._build_api_evidence_text(candidate_name, api_species_code, api_region_evidence),
                        source_mode="api",
                        species_code=api_species_code,
                        region_evidence=api_region_evidence,
                    )
                raise EbirdLookupError("redirected_to_login")
            return None
        if response.status_code >= 400:
            return None

        identification_text = self._extract_identification_text(response.text)
        if not identification_text:
            if api_species_code:
                return EbirdLookupResult(
                    source_candidate=candidate_name,
                    species_page_url=species_url,
                    identification_text=self._build_api_evidence_text(candidate_name, api_species_code, api_region_evidence),
                    source_mode="api",
                    species_code=api_species_code,
                    region_evidence=api_region_evidence,
                )
            return None

        return EbirdLookupResult(
            source_candidate=candidate_name,
            species_page_url=species_url,
            identification_text=identification_text,
            source_mode=source_mode,
            species_code=api_species_code,
            region_evidence=api_region_evidence,
        )

    def _build_api_evidence_text(self, candidate_name: str, species_code: str, region_evidence: Dict[str, int]) -> str:
        parts = [f"{k}:{v}" for k, v in region_evidence.items()]
        regions_line = ", ".join(parts) if parts else "no recent records"
        return (
            f"API evidence for candidate {candidate_name}. "
            f"eBird species code: {species_code}. "
            f"Recent observations by configured regions: {regions_line}."
        )

    def _slugify(self, name: str) -> str:
        # eBird slugs are typically lowercase and compact (best-effort).
        normalized = re.sub(r"[^a-zA-Z0-9]+", "", name).lower()
        if not normalized:
            return quote(name.lower().strip())
        return normalized

    def _extract_identification_text(self, html: str) -> str:
        if not html:
            return ""
        lowered = html.lower()
        marker = "identification"
        idx = lowered.find(marker)
        if idx == -1:
            return ""

        window = html[idx : idx + 8000]
        # Remove scripts/styles/tags and compress whitespace.
        window = re.sub(r"(?is)<script.*?>.*?</script>", " ", window)
        window = re.sub(r"(?is)<style.*?>.*?</style>", " ", window)
        text = re.sub(r"(?is)<[^>]+>", " ", window)
        text = unescape(text)
        text = re.sub(r"\s+", " ", text).strip()
        # Keep concise chunk around the marker.
        id_pos = text.lower().find("identification")
        if id_pos >= 0:
            text = text[id_pos:]
        return text[:1200]

