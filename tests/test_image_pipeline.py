import io

from PIL import Image

import core.image_pipeline as image_pipeline
from core.image_pipeline import ImageValidationError, prepare_image


def _make_image_bytes(size=(1200, 800), fmt="PNG"):
    image = Image.new("RGB", size=size, color=(120, 80, 40))
    output = io.BytesIO()
    image.save(output, format=fmt)
    return output.getvalue()


def test_prepare_image_returns_data_url_and_hash():
    raw = _make_image_bytes()
    result = prepare_image(raw, max_upload_mb=8, max_image_side=1024)
    assert result["data_url"].startswith("data:image/webp;base64,")
    assert len(result["hash"]) == 64


def test_prepare_image_falls_back_to_jpeg_when_webp_unavailable(monkeypatch):
    raw = _make_image_bytes()

    def _raise_webp_error(_image):
        raise OSError("WEBP codec not available")

    monkeypatch.setattr(image_pipeline, "_encode_webp", _raise_webp_error)

    result = prepare_image(raw, max_upload_mb=8, max_image_side=1024)
    assert result["data_url"].startswith("data:image/jpeg;base64,")
    assert len(result["hash"]) == 64


def test_prepare_image_rejects_huge_source():
    raw = b"x" * (51 * 1024 * 1024)
    try:
        prepare_image(raw, max_upload_mb=8, max_image_side=1024)
    except ImageValidationError as exc:
        assert "under 50MB" in str(exc)
    else:
        raise AssertionError("Expected ImageValidationError for huge source file")
