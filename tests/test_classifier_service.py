from core.classifier_service import ClassifierService
from core.ebird_api_client import EbirdApiSpeciesMatch


class _FakeClient:
    def __init__(self, output: str):
        self._output = output

    def classify_bird(self, image_data_url: str, prompt: str) -> str:
        return self._output

    def classify_bird_from_text(self, text_content: str, prompt: str) -> str:  # noqa: ARG002
        return self._output


def test_classifier_parses_embedded_json():
    output = """Sure, here is JSON:
{
  "predictions": [{"species":"A","common_name":"B","confidence":0.8,"features":["f1"]}],
  "reasoning":"ok",
  "is_bird": true
}
"""
    service = ClassifierService(_FakeClient(output), top_k=3)
    result = service.classify("data:image/jpeg;base64,abc")
    assert result["is_bird"] is True
    assert result["predictions"][0]["common_name"] == "B"
    assert result["predictions"][0]["confidence"] == 0.8


def test_classifier_recovers_candidates_from_prose_when_no_json():
    output = """Based on the image:
* **Mandarin Duck**
* Wood Duck
Top 1: Northern Pintail (best match)
"""
    service = ClassifierService(_FakeClient(output), top_k=3)
    result = service.classify("data:image/jpeg;base64,abc")
    assert result["predictions"]
    names = [p["common_name"] for p in result["predictions"]]
    assert "Mandarin Duck" in names


def test_classifier_normalizes_non_schema_payload():
    output = """
{
  "type": "bird",
  "name": "Mandarin Duck",
  "description": "duck on shore",
  "habitat": "wetland"
}
"""
    service = ClassifierService(_FakeClient(output), top_k=3)
    result = service.classify("data:image/jpeg;base64,abc")
    assert result["is_bird"] is True
    assert result["predictions"][0]["common_name"] == "Mandarin Duck"
    assert result["predictions"][0]["confidence"] == 0.7


def test_classifier_ebird_ocr_mode_parses_json():
    output = """
{
  "predictions": [{"species":"Blue Flycatcher","common_name":"Hainan Blue Flycatcher","confidence":0.76,"features":["blue upperparts","orange throat"]}],
  "reasoning":"The text clues match a blue flycatcher profile.",
  "ebird_identification_text":"Medium-sized flycatcher with deep blue upperparts and orange throat.",
  "is_bird": true
}
"""
    service = ClassifierService(_FakeClient(output), top_k=3)
    result = service.classify_ebird_ocr("data:image/webp;base64,abc")
    assert result["is_bird"] is True
    assert result["predictions"][0]["common_name"] == "Hainan Blue Flycatcher"
    assert result["ebird_identification_text"].startswith("Medium-sized flycatcher")


def test_fetch_ebird_recent_fallbacks_across_top_candidates():
    match = EbirdApiSpeciesMatch(
        sci_name="Cyornis whitei",
        com_name="Hainan Blue Flycatcher",
        species_code="hilfly1",
    )

    class _FakeApi:
        enabled = True

        def find_species(self, name: str):
            if name == "Top1 Wrong Name":
                return None
            if name == "Cyornis whitei":
                return match
            return None

        def region_recent_observations(self, species_code, regions, max_per_region):  # noqa: ARG002
            return {"HK": [{"obsDt": "2024-01-01", "locName": "Tai Po", "lat": None, "lng": None, "howMany": 2}]}

    service = ClassifierService(
        _FakeClient("{}"),
        top_k=3,
        ebird_api=_FakeApi(),
        ebird_regions=["HK"],
        ebird_recent_max=5,
    )
    result = service.fetch_ebird_recent_for_predictions(
        [
            {"species": "Top1 Wrong Name", "common_name": "Top1 Wrong Name", "confidence": 0.9},
            {"species": "Cyornis whitei", "common_name": "Hainan Blue Flycatcher", "confidence": 0.7},
        ]
    )
    assert result["status"] == "ok"
    assert result["species_code"] == "hilfly1"
    assert result["source_candidate"] == "Cyornis whitei"
    assert result["recent_by_region"]["HK"][0]["locName"] == "Tai Po"


def test_rerank_predictions_with_region_evidence_boosts_matched_candidate():
    service = ClassifierService(_FakeClient("{}"), top_k=3, ebird_api=None)
    preds = [
        {"species": "Unknown bird", "common_name": "Unknown bird", "confidence": 0.82, "features": []},
        {"species": "Cyornis whitei", "common_name": "Hainan Blue Flycatcher", "confidence": 0.76, "features": []},
    ]
    ebird_recent = {
        "status": "ok",
        "sci_name": "Cyornis whitei",
        "com_name": "Hainan Blue Flycatcher",
        "source_candidate": "Cyornis whitei",
        "recent_by_region": {"HK": [{"locName": "Tai Po"}], "CN": [{"locName": "Yunnan"}]},
    }
    reranked = service.rerank_predictions_with_ebird_recent(preds, ebird_recent)
    assert reranked[0]["common_name"] == "Hainan Blue Flycatcher"
    assert reranked[0]["rerank_score"] > reranked[1]["rerank_score"]
