from types import SimpleNamespace

from core.ebird_lookup import EbirdLookupService


def test_lookup_extracts_identification_text():
    service = EbirdLookupService(api_token="t", regions=["HK", "CN"])
    html = """
    <html><body>
      <h2>Identification</h2>
      <p>Medium-sized confusing flycatcher with deep blue upperparts and orange throat.</p>
    </body></html>
    """

    def fake_get(_url, timeout, allow_redirects):  # noqa: ARG001
        return SimpleNamespace(status_code=200, headers={}, text=html)

    service._session.get = fake_get  # type: ignore[attr-defined]
    service._api_client.find_species = lambda _name: SimpleNamespace(species_code="hilfly1")  # type: ignore[attr-defined]
    service._api_client.region_evidence = lambda _code, _regions: {"HK": 3, "CN": 11}  # type: ignore[attr-defined]
    result = service.lookup_identification(["Cyornis whitei"])
    assert result is not None
    assert "Identification" in result.identification_text
    assert "flycatcher" in result.identification_text.lower()
    assert result.source_mode == "mixed"
    assert result.species_code == "hilfly1"
    assert result.region_evidence == {"HK": 3, "CN": 11}


def test_lookup_returns_none_on_login_redirect():
    service = EbirdLookupService()

    def fake_get(_url, timeout, allow_redirects):  # noqa: ARG001
        return SimpleNamespace(status_code=302, headers={"Location": "https://secure.birds.cornell.edu/login"}, text="")

    service._session.get = fake_get  # type: ignore[attr-defined]
    try:
        result = service.lookup_identification(["Cyornis whitei"])
    except Exception as exc:  # noqa: BLE001
        assert "redirected_to_login" in str(exc)
    else:
        assert result is None
