import json
import re
from typing import Any, Dict, List

from core.deepseek_client import DeepSeekClient
from core.ebird_api_client import EbirdApiClient, EbirdApiError

_STOPWORDS = frozenset(
    {
        "the",
        "image",
        "this",
        "that",
        "bird",
        "species",
        "based",
        "here",
        "json",
        "predictions",
        "reasoning",
        "top",
        "possible",
        "likely",
        "appears",
        "shows",
        "photo",
        "picture",
        "following",
        "three",
        "list",
    }
)

PROMPT_TEMPLATE = """
You are an expert bird field guide. Look at the photo and identify the bird.

CRITICAL OUTPUT RULES:
- Reply with ONE JSON object only.
- The first character of your reply MUST be "{{" and the last MUST be "}}".
- Do NOT write any text before or after the JSON. No markdown, no code fences, no bullet lists outside JSON.
- Use English common names in "common_name" (e.g. "Mandarin Duck").

JSON schema (exact keys):
{{
  "predictions": [
    {{
      "species": "string (English common name or Latin if unsure)",
      "common_name": "string (English common name)",
      "confidence": 0.0,
      "features": ["short visible field marks"]
    }}
  ],
  "reasoning": "one short sentence why",
  "is_bird": true,
  "hk_rarity_note": "if rare in Hong Kong, explicitly warn; otherwise short note",
  "hk_common_places": ["short place names in Hong Kong where this bird is commonly seen"]
}}

Rules:
- Return exactly up to {top_k} predictions, best guess first.
- confidence between 0 and 1 (sum does not need to be 1).
- If the subject is clearly not a bird, set is_bird false and predictions [].
- If uncertain, still output best-effort candidates with lower confidence.
- Include habitat/behavior clues in features when possible (eye-ring, bill shape, plumage colors).
""".strip()

EBIRD_OCR_PROMPT_TEMPLATE = """
You are an expert bird field guide.
The uploaded image is an eBird screenshot and may contain text descriptions of a bird.
Focus on reading the bird description text in the screenshot and infer the likely species.

CRITICAL OUTPUT RULES:
- Reply with ONE JSON object only.
- The first character of your reply MUST be "{{" and the last MUST be "}}".
- Do NOT write any text before or after the JSON. No markdown, no code fences, no bullet lists outside JSON.
- Use English common names in "common_name".

JSON schema (exact keys):
{{
  "predictions": [
    {{
      "species": "string (English common name or Latin if unsure)",
      "common_name": "string (English common name)",
      "confidence": 0.0,
      "features": ["short clues inferred from the text description"]
    }}
  ],
  "reasoning": "one short sentence based on the screenshot text",
  "ebird_identification_text": "short extracted identification text from the screenshot if visible",
  "is_bird": true,
  "hk_rarity_note": "if rare in Hong Kong, explicitly warn; otherwise short note",
  "hk_common_places": ["short place names in Hong Kong where this bird is commonly seen"]
}}

Rules:
- Return exactly up to {top_k} predictions, best guess first.
- confidence between 0 and 1 (sum does not need to be 1).
- If the screenshot text is not about a bird, set is_bird false and predictions [].
- If text is partial or blurry, still output best-effort candidates with lower confidence.
""".strip()

class ClassifierService:
    def __init__(
        self,
        client: DeepSeekClient,
        top_k: int,
        ebird_api: EbirdApiClient | None = None,
        ebird_regions: List[str] | None = None,
        ebird_recent_max: int = 10,
    ) -> None:
        self._client = client
        self._top_k = top_k
        self._ebird_api = ebird_api
        self._ebird_regions = [x.strip().upper() for x in (ebird_regions or ["HK", "CN"]) if x.strip()]
        self._ebird_recent_max = max(1, min(ebird_recent_max, 100))

    def _extract_predictions_from_prose(self, raw: str) -> List[Dict[str, Any]]:
        """When the model ignores JSON, pull likely English bird names from prose."""
        text = raw or ""
        candidates: List[str] = []
        seen: set[str] = set()

        def push(name: str) -> None:
            name = re.sub(r"\s+", " ", name).strip(" -•\t*")
            if len(name) < 3 or len(name) > 100:
                return
            low = name.lower()
            if low in _STOPWORDS:
                return
            if name in seen:
                return
            seen.add(name)
            candidates.append(name)

        for m in re.finditer(r"\*\*([A-Za-z][A-Za-z0-9 \-']{2,78})\*\*", text):
            push(m.group(1))

        for m in re.finditer(r"(?m)^\s*[\*\-]\s*(.+)$", text):
            line = m.group(1).strip()
            line = re.sub(r"^\*+\s*", "", line)
            line = re.sub(r"^(?:Top\s*\d+\s*[:.]|\d+\.\s*)", "", line, flags=re.IGNORECASE)
            if ":" in line[:50]:
                parts = line.split(":", 1)
                rest = parts[1].strip()
                if len(rest) > 3:
                    rest = re.sub(r"\([^)]*\)", "", rest).strip()
                    push(rest)
            elif len(line) > 3:
                line = re.sub(r"\([^)]*\)", "", line).strip()
                push(line)

        for m in re.finditer(r"(?i)Top\s*\d+\s*:\s*([^(|\n]{3,80})", text):
            push(m.group(1).strip())

        out: List[Dict[str, Any]] = []
        for i, name in enumerate(candidates[: self._top_k]):
            conf = max(0.35, min(0.9, 0.88 - i * 0.1))
            out.append(
                {
                    "species": name,
                    "common_name": name,
                    "confidence": conf,
                    "features": [],
                }
            )
        return out

    def _classify_with_prompt(self, image_data_url: str, prompt_template: str) -> Dict[str, Any]:
        raw = self._client.classify_bird(image_data_url=image_data_url, prompt=prompt_template.format(top_k=self._top_k))
        try:
            parsed = self._normalize_schema(self._parse_json(raw))
        except ValueError:
            fallback = self._extract_predictions_from_prose(raw)
            return {
                "is_bird": True,
                "predictions": fallback,
                "reasoning": raw.strip(),
                "raw_response": raw,
                "parse_note": "non_json_response" if not fallback else "recovered_from_prose",
            }
        predictions = parsed.get("predictions") or []

        cleaned_predictions: List[Dict[str, Any]] = []
        for item in predictions[: self._top_k]:
            try:
                confidence = float(item.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))
            cleaned_predictions.append(
                {
                    "species": str(item.get("species", "")),
                    "common_name": str(item.get("common_name", "")),
                    "confidence": confidence,
                    "features": [str(x) for x in (item.get("features") or [])][:5],
                }
            )

        cleaned_predictions.sort(key=lambda x: x["confidence"], reverse=True)
        if not cleaned_predictions:
            cleaned_predictions = self._extract_predictions_from_prose(raw)
        reasoning = str(parsed.get("reasoning", "")).strip()
        if not reasoning and raw:
            reasoning = raw.strip()[:2000]
        result: Dict[str, Any] = {
            "is_bird": bool(parsed.get("is_bird", True)),
            "predictions": cleaned_predictions,
            "reasoning": reasoning,
            "raw_response": raw,
            "ebird_identification_text": str(parsed.get("ebird_identification_text", "")).strip(),
            "hk_rarity_note": str(parsed.get("hk_rarity_note", "")).strip(),
            "hk_common_places": [str(x).strip() for x in (parsed.get("hk_common_places") or []) if str(x).strip()],
        }
        if not predictions and cleaned_predictions:
            result["parse_note"] = "filled_from_prose"
        return result

    def classify(self, image_data_url: str) -> Dict[str, Any]:
        return self._classify_with_prompt(image_data_url=image_data_url, prompt_template=PROMPT_TEMPLATE)

    def classify_ebird_ocr(self, image_data_url: str) -> Dict[str, Any]:
        return self._classify_with_prompt(image_data_url=image_data_url, prompt_template=EBIRD_OCR_PROMPT_TEMPLATE)

    def fetch_ebird_recent_for_predictions(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve species via eBird taxonomy API and fetch recent observations per region only."""
        if not self._ebird_api or not self._ebird_api.enabled:
            return {"status": "no_token"}

        match = None
        source_candidate = ""

        for item in predictions[: self._top_k]:
            species = str(item.get("species", "")).strip()
            common = str(item.get("common_name", "")).strip()
            labels: List[str] = []
            if species:
                labels.append(species)
            if common and common.lower() != species.lower():
                labels.append(common)
            for label in labels:
                try:
                    found = self._ebird_api.find_species(label)
                except EbirdApiError as exc:
                    return {"status": "api_error", "error": str(exc)}
                if found:
                    match = found
                    source_candidate = label
                    break
            if match:
                break

        if not match:
            return {"status": "no_match"}

        try:
            recent_by_region = self._ebird_api.region_recent_observations(
                match.species_code, self._ebird_regions, self._ebird_recent_max
            )
        except EbirdApiError as exc:
            return {"status": "api_error", "error": str(exc)}

        species_url = f"https://ebird.org/species/{match.species_code}"
        return {
            "status": "ok",
            "species_code": match.species_code,
            "sci_name": match.sci_name,
            "com_name": match.com_name,
            "source_candidate": source_candidate,
            "species_page_url": species_url,
            "recent_by_region": recent_by_region,
        }

    def rerank_predictions_with_ebird_recent(
        self, predictions: List[Dict[str, Any]], ebird_recent: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Re-rank model predictions by adding light region-evidence weighting."""
        if not predictions:
            return predictions
        if not ebird_recent or ebird_recent.get("status") != "ok":
            return predictions

        recent_by_region = ebird_recent.get("recent_by_region") or {}
        observed_regions = sum(1 for rows in recent_by_region.values() if rows)
        # Suggested baseline: +0.12 if at least one configured region has recent obs.
        region_bonus = 0.12 if observed_regions > 0 else -0.05
        if observed_regions > 1:
            region_bonus = min(0.20, region_bonus + 0.04 * (observed_regions - 1))

        com_name = str(ebird_recent.get("com_name", "")).strip().lower()
        sci_name = str(ebird_recent.get("sci_name", "")).strip().lower()
        source_candidate = str(ebird_recent.get("source_candidate", "")).strip().lower()
        match_labels = {x for x in [com_name, sci_name, source_candidate] if x}

        reranked: List[Dict[str, Any]] = []
        for item in predictions:
            model_confidence = max(0.0, min(1.0, float(item.get("confidence", 0.0))))
            pred_species = str(item.get("species", "")).strip().lower()
            pred_common = str(item.get("common_name", "")).strip().lower()
            is_evidence_match = pred_species in match_labels or pred_common in match_labels

            adjusted_bonus = region_bonus if is_evidence_match else (-0.02 if region_bonus > 0 else 0.0)
            rerank_score = max(0.0, min(1.0, model_confidence + adjusted_bonus))

            updated = dict(item)
            updated["model_confidence"] = model_confidence
            updated["region_bonus"] = adjusted_bonus
            updated["rerank_score"] = rerank_score
            updated["evidence_matched"] = is_evidence_match
            # Use re-ranked score as display confidence in Top ordering.
            updated["confidence"] = rerank_score
            reranked.append(updated)

        reranked.sort(key=lambda x: float(x.get("rerank_score", x.get("confidence", 0.0))), reverse=True)
        return reranked[: self._top_k]

    def _parse_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        if not text:
            raise ValueError("Model response is empty.")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
            if fenced:
                return json.loads(fenced.group(1))
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise ValueError("Model response is not valid JSON.")
            return json.loads(match.group(0))

    def _normalize_schema(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        # Normal case: model already follows required schema.
        if isinstance(parsed.get("predictions"), list):
            if "reasoning" not in parsed:
                parsed["reasoning"] = str(parsed.get("description", "")).strip()
            if "is_bird" not in parsed:
                parsed["is_bird"] = True
            if "hk_rarity_note" not in parsed:
                parsed["hk_rarity_note"] = str(parsed.get("rarity_note", "")).strip()
            if "hk_common_places" not in parsed:
                places: List[str] = []
                # Backward-compatible mapping from old fields.
                hung_hom = str(parsed.get("hong_hom_commonness") or parsed.get("hung_hom_commonness") or "").strip()
                wetland = str(parsed.get("hk_wetland_park_commonness") or parsed.get("wetland_park_commonness") or "").strip()
                if hung_hom:
                    places.append(f"紅磡（{hung_hom}）")
                if wetland:
                    places.append(f"香港濕地公園（{wetland}）")
                parsed["hk_common_places"] = places
            return parsed

        # Fallback for alternate response shapes from some vision models.
        name = str(parsed.get("name") or parsed.get("common_name") or parsed.get("species") or "").strip()
        species = str(parsed.get("species") or name).strip()
        reasoning = str(parsed.get("reasoning") or parsed.get("description") or "").strip()
        bird_type = str(parsed.get("type", "")).strip().lower()
        is_bird = bool(parsed.get("is_bird", bird_type in {"bird", ""} or bool(name)))

        features: List[str] = []
        if parsed.get("habitat"):
            features.append(f"habitat: {parsed['habitat']}")
        if parsed.get("diet"):
            features.append(f"diet: {parsed['diet']}")
        if parsed.get("conservation_status"):
            features.append(f"status: {parsed['conservation_status']}")

        predictions: List[Dict[str, Any]] = []
        if name or species:
            predictions.append(
                {
                    "species": species,
                    "common_name": name,
                    "confidence": 0.7,
                    "features": features[:5],
                }
            )

        return {
            "predictions": predictions,
            "reasoning": reasoning,
            "ebird_identification_text": "",
            "is_bird": is_bird,
            "hk_rarity_note": "",
            "hk_common_places": [],
        }
