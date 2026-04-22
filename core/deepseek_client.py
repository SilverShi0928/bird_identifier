import time
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests


class DeepSeekClientError(Exception):
    pass


class DeepSeekClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: int,
        retry_count: int,
        *,
        openrouter_provider_ignore: Optional[List[str]] = None,
        openrouter_http_referer: str = "",
        openrouter_title: str = "",
    ) -> None:
        self._base_url = base_url
        self._api_key = api_key
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._retry_count = retry_count
        self._openrouter_provider_ignore = openrouter_provider_ignore or []
        self._openrouter_http_referer = openrouter_http_referer.strip()
        self._openrouter_title = openrouter_title.strip()

    def _is_openrouter(self) -> bool:
        return "openrouter.ai" in self._base_url

    def _request_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if self._is_openrouter():
            referer = self._openrouter_http_referer or "https://github.com/local/bird_identifier"
            title = self._openrouter_title or "雀鳥辨識"
            headers.setdefault("HTTP-Referer", referer)
            # Some Chat Completions gateways expect this attribution header (plus `X-Title` fallback).
            safe_title = self._header_safe_latin1(title)
            headers.setdefault("X-OpenRouter-Title", safe_title)
            headers.setdefault("X-Title", safe_title)
        return headers

    @staticmethod
    def _header_safe_latin1(value: str) -> str:
        """
        `requests` enforces latin-1 for HTTP header values.
        Encode non-latin text into an ASCII-safe representation.
        """
        try:
            value.encode("latin-1")
            return value
        except UnicodeEncodeError:
            return quote(value, safe="-_.~ ")

    def _merge_openrouter_provider(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self._is_openrouter() or not self._openrouter_provider_ignore:
            return payload
        provider = dict(payload.get("provider") or {})
        merged_ignore = [str(x).strip().lower() for x in (provider.get("ignore") or []) if str(x).strip()]
        merged_ignore.extend(self._openrouter_provider_ignore)
        deduped: List[str] = []
        seen: set[str] = set()
        for slug in merged_ignore:
            if slug in seen:
                continue
            seen.add(slug)
            deduped.append(slug)
        provider["ignore"] = deduped
        payload["provider"] = provider
        return payload

    def classify_bird(self, image_data_url: str, prompt: str) -> str:
        if not self._api_key:
            raise DeepSeekClientError(
                "缺少辨識 API 金鑰。請於 `.env` 設定（範例見 `.env.example`）。"
            )

        payload: Dict[str, Any] = self._merge_openrouter_provider(
            {
                "model": self._model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                        ],
                    }
                ],
                "temperature": 0,
            }
        )
        headers = self._request_headers()

        errors: List[str] = []
        for attempt in range(self._retry_count + 1):
            try:
                response = requests.post(
                    f"{self._base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self._timeout_seconds,
                )
                if response.status_code >= 400:
                    raise DeepSeekClientError(
                        f"辨識 API 回傳錯誤 {response.status_code}：{response.text[:400]}"
                    )

                body = response.json()
                return body["choices"][0]["message"]["content"]
            except (requests.RequestException, KeyError, ValueError, DeepSeekClientError) as exc:
                errors.append(str(exc))
                if attempt >= self._retry_count:
                    break
                time.sleep(2 ** attempt)

        raise DeepSeekClientError("多次重試後仍無法完成辨識：" + " | ".join(errors))

    def classify_bird_from_text(self, text_content: str, prompt: str) -> str:
        if not self._api_key:
            raise DeepSeekClientError(
                "缺少辨識 API 金鑰。請於 `.env` 設定（範例見 `.env.example`）。"
            )
        payload: Dict[str, Any] = self._merge_openrouter_provider(
            {
                "model": self._model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "text", "text": f"eBird Identification text:\n{text_content}"},
                        ],
                    }
                ],
                "temperature": 0,
            }
        )
        headers = self._request_headers()

        errors: List[str] = []
        for attempt in range(self._retry_count + 1):
            try:
                response = requests.post(
                    f"{self._base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self._timeout_seconds,
                )
                if response.status_code >= 400:
                    raise DeepSeekClientError(
                        f"辨識 API 回傳錯誤 {response.status_code}：{response.text[:400]}"
                    )

                body = response.json()
                return body["choices"][0]["message"]["content"]
            except (requests.RequestException, KeyError, ValueError, DeepSeekClientError) as exc:
                errors.append(str(exc))
                if attempt >= self._retry_count:
                    break
                time.sleep(2 ** attempt)

        raise DeepSeekClientError("多次重試後仍無法完成辨識：" + " | ".join(errors))
