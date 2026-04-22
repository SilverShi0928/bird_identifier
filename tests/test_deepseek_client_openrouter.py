from unittest.mock import MagicMock, patch

from core.deepseek_client import DeepSeekClient


@patch("core.deepseek_client.requests.post")
def test_openrouter_merges_provider_ignore_and_headers(mock_post: MagicMock) -> None:
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"choices": [{"message": {"content": "ok"}}]}

    client = DeepSeekClient(
        base_url="https://openrouter.ai/api/v1",
        api_key="k",
        model="m",
        timeout_seconds=5,
        retry_count=0,
        openrouter_provider_ignore=["novita"],
        openrouter_http_referer="https://example.test/app",
        openrouter_title="Example App",
    )
    out = client.classify_bird("data:image/jpeg;base64,abc", "prompt")

    assert out == "ok"
    kwargs = mock_post.call_args.kwargs
    assert kwargs["json"]["provider"]["ignore"] == ["novita"]
    assert kwargs["headers"]["HTTP-Referer"] == "https://example.test/app"
    assert kwargs["headers"]["X-OpenRouter-Title"] == "Example App"


@patch("core.deepseek_client.requests.post")
def test_non_openrouter_skips_provider_routing(mock_post: MagicMock) -> None:
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"choices": [{"message": {"content": "ok"}}]}

    client = DeepSeekClient(
        base_url="https://api.deepseek.com/v1",
        api_key="k",
        model="m",
        timeout_seconds=5,
        retry_count=0,
        openrouter_provider_ignore=["novita"],
    )
    client.classify_bird("data:image/jpeg;base64,abc", "prompt")

    kwargs = mock_post.call_args.kwargs
    assert "provider" not in kwargs["json"]
    assert "X-OpenRouter-Title" not in kwargs["headers"]


@patch("core.deepseek_client.requests.post")
def test_openrouter_title_is_header_safe_when_non_latin(mock_post: MagicMock) -> None:
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"choices": [{"message": {"content": "ok"}}]}

    client = DeepSeekClient(
        base_url="https://openrouter.ai/api/v1",
        api_key="k",
        model="m",
        timeout_seconds=5,
        retry_count=0,
        openrouter_title="雀鳥辨識",
    )
    client.classify_bird("data:image/jpeg;base64,abc", "prompt")

    kwargs = mock_post.call_args.kwargs
    # Percent-encoded UTF-8, safe for latin-1 constrained HTTP headers.
    assert kwargs["headers"]["X-OpenRouter-Title"] == "%E9%9B%80%E9%B3%A5%E8%BE%A8%E8%AD%98"
