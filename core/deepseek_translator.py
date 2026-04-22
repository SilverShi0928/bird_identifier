import json
import re
from typing import Any, Dict, List

import requests


class DeepSeekTranslator:
    def __init__(self, api_key: str, base_url: str, model: str, timeout_seconds: int) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_seconds = timeout_seconds

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    def translate_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            return {
                "reasoning_zh": "",
                "predictions_zh": [],
                "hk_rarity_note_zh": "",
                "hk_common_places_zh": [],
            }

        predictions: List[Dict[str, Any]] = result.get("predictions") or []
        lines: List[str] = []
        for idx, pred in enumerate(predictions, start=1):
            label = pred.get("common_name") or pred.get("species") or "Unknown"
            confidence = float(pred.get("confidence", 0.0))
            lines.append(f"Top {idx}: {label} ({confidence:.0%})")

        payload_text = (
            "Translate the following bird identification result into Traditional Chinese.\n"
            "Return JSON only with this schema:\n"
            "{\n"
            '  "reasoning_zh": "string",\n'
            '  "predictions_zh": [\n'
            "    {\n"
            '      "rank": 1,\n'
            '      "label_zh": "string",\n'
            '      "features_zh": ["string"]\n'
            "    }\n"
            "  ],\n"
            '  "hk_rarity_note_zh": "string",\n'
            '  "hk_common_places_zh": ["string"]\n'
            "}\n"
            "No markdown.\n\n"
            f"Reasoning:\n{result.get('reasoning', '')}\n\n"
            "Predictions:\n"
            + ("\n".join(lines) if lines else "No predictions")
            + "\n\n"
            + f"Hong Kong rarity note:\n{result.get('hk_rarity_note', '')}\n\n"
            + "Hong Kong common places:\n"
            + ", ".join(result.get("hk_common_places") or [])
        )

        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": payload_text}],
            "temperature": 0.2,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            f"{self._base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self._timeout_seconds,
        )
        if response.status_code >= 400:
            return {
                "reasoning_zh": "",
                "predictions_zh": [],
                "hk_rarity_note_zh": "",
                "hk_common_places_zh": [],
            }
        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            parsed = self._parse_json_content(content)
            reasoning_zh = str(parsed.get("reasoning_zh", "")).strip()
            hk_rarity_note_zh = str(parsed.get("hk_rarity_note_zh", "")).strip()
            hk_common_places_zh = [str(x).strip() for x in (parsed.get("hk_common_places_zh") or []) if str(x).strip()]
            translated_predictions = parsed.get("predictions_zh") or []
            cleaned_predictions = []
            for item in translated_predictions:
                cleaned_predictions.append(
                    {
                        "rank": int(item.get("rank", 0)),
                        "label_zh": str(item.get("label_zh", "")).strip(),
                        "features_zh": [str(x).strip() for x in (item.get("features_zh") or []) if str(x).strip()],
                    }
                )
            if not reasoning_zh and isinstance(content, str):
                # Fallback: upstream sometimes returns plain Chinese text instead of JSON.
                reasoning_zh = content.strip()

            if not cleaned_predictions:
                cleaned_predictions = self._fallback_translate_predictions(result)

            return {
                "reasoning_zh": reasoning_zh,
                "predictions_zh": cleaned_predictions,
                "hk_rarity_note_zh": hk_rarity_note_zh,
                "hk_common_places_zh": hk_common_places_zh,
            }
        except Exception:
            return {
                "reasoning_zh": "",
                "predictions_zh": self._fallback_translate_predictions(result),
                "hk_rarity_note_zh": "",
                "hk_common_places_zh": [],
            }

    def _parse_json_content(self, content: Any) -> Dict[str, Any]:
        if isinstance(content, dict):
            return content
        if not isinstance(content, str):
            return {}

        text = content.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            fenced_start = text.find("{")
            fenced_end = text.rfind("}")
            if fenced_start >= 0 and fenced_end > fenced_start:
                try:
                    return json.loads(text[fenced_start : fenced_end + 1])
                except json.JSONDecodeError:
                    return {}
            return {}

    def _fallback_translate_predictions(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        predictions: List[Dict[str, Any]] = result.get("predictions") or []
        if not predictions:
            return []

        lines: List[str] = []
        for idx, pred in enumerate(predictions, start=1):
            label = pred.get("common_name") or pred.get("species") or "Unknown"
            features = ", ".join(pred.get("features") or [])
            lines.append(f"Top {idx}: {label}; features: {features}")

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Translate each line into Traditional Chinese. Keep one line per item.\n"
                        "Format strictly as:\n"
                        "Top N: <中文名稱> | 特徵: <中文特徵>\n\n"
                        + "\n".join(lines)
                    ),
                }
            ],
            "temperature": 0.2,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            f"{self._base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self._timeout_seconds,
        )
        if response.status_code >= 400:
            return []
        try:
            content = response.json()["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                return []
            out: List[Dict[str, Any]] = []
            for line in content.splitlines():
                line = line.strip()
                if not line.startswith("Top "):
                    continue
                # Example: Top 1: 金雉 | 特徵: 長尾、鮮艷羽色
                rank = 0
                label_zh = ""
                features_zh: List[str] = []
                try:
                    left, right = (line.split("|", 1) + [""])[:2]
                    rank_part, label_part = left.split(":", 1)
                    rank = int(rank_part.replace("Top", "").strip())
                    label_zh = label_part.strip()
                    if "特徵" in right:
                        features_text = right.split(":", 1)[1].strip()
                        features_zh = [x.strip() for x in re.split(r"[、,;/]", features_text) if x.strip()]
                except Exception:
                    continue
                out.append({"rank": rank, "label_zh": label_zh, "features_zh": features_zh})
            return out
        except Exception:
            return []
