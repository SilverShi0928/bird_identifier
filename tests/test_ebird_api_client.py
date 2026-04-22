from types import SimpleNamespace

from core.ebird_api_client import EbirdApiClient


def test_find_species_returns_best_match():
    client = EbirdApiClient(api_token="token")

    rows = [
        {"sciName": "Other name", "comName": "Other", "speciesCode": "other1"},
        {"sciName": "Cyornis whitei", "comName": "Hainan Blue Flycatcher", "speciesCode": "hilfly1"},
    ]

    def fake_get(_url, params=None, timeout=None):  # noqa: ARG001
        return SimpleNamespace(status_code=200, json=lambda: rows)

    client._session.get = fake_get  # type: ignore[attr-defined]
    match = client.find_species("Cyornis whitei")
    assert match is not None
    assert match.species_code == "hilfly1"
    assert match.sci_name == "Cyornis whitei"


def test_recent_observations_maps_fields():
    client = EbirdApiClient(api_token="token")
    payload = [
        {
            "obsDt": "2024-01-01 08:00",
            "locName": "Victoria Park",
            "lat": 22.28,
            "lng": 114.19,
            "howMany": 3,
            "subnational1Code": "HK-KKC",
        }
    ]

    def fake_get(url, params=None, timeout=None):  # noqa: ARG001
        if "/HK/recent/" in url:
            return SimpleNamespace(status_code=200, json=lambda: payload)
        return SimpleNamespace(status_code=404, json=lambda: [])

    client._session.get = fake_get  # type: ignore[attr-defined]
    rows = client.recent_observations("HK", "eurspa1", max_rows=10)
    assert len(rows) == 1
    assert rows[0]["locName"] == "Victoria Park"
    assert rows[0]["obsDt"] == "2024-01-01 08:00"


def test_region_evidence_counts_recent_records():
    client = EbirdApiClient(api_token="token")

    def fake_get(url, params=None, timeout=None):  # noqa: ARG001
        if "/HK/recent/" in url:
            return SimpleNamespace(status_code=200, json=lambda: [{"id": 1}, {"id": 2}])
        if "/CN/recent/" in url:
            return SimpleNamespace(status_code=200, json=lambda: [{"id": 1}])
        return SimpleNamespace(status_code=404, json=lambda: [])

    client._session.get = fake_get  # type: ignore[attr-defined]
    evidence = client.region_evidence("hilfly1", ["HK", "CN"])
    assert evidence == {"HK": 2, "CN": 1}
