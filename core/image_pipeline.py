import base64
import hashlib
import io
from typing import Dict

from PIL import Image, UnidentifiedImageError
try:
    from pillow_heif import register_heif_opener
except ImportError:  # pragma: no cover - optional dependency
    register_heif_opener = None


# Enable HEIC/HEIF decoding via Pillow.
if register_heif_opener is not None:
    register_heif_opener()


class ImageValidationError(Exception):
    pass


def _encode_webp(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="WEBP", quality=82, method=6)
    return output.getvalue()


def _encode_jpeg(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=90)
    return output.getvalue()


def prepare_image(file_bytes: bytes, max_upload_mb: int, max_image_side: int) -> Dict[str, str]:
    max_bytes = max_upload_mb * 1024 * 1024
    # Keep a generous hard cap to avoid memory abuse, but allow large source
    # images to be downscaled/compressed before enforcing max_upload_mb.
    hard_input_cap_bytes = 50 * 1024 * 1024
    if len(file_bytes) > hard_input_cap_bytes:
        raise ImageValidationError("Source file too large. Please use an image under 50MB.")

    try:
        image = Image.open(io.BytesIO(file_bytes))
    except UnidentifiedImageError as exc:
        raise ImageValidationError("Unsupported image format.") from exc

    image = image.convert("RGB")
    image.thumbnail((max_image_side, max_image_side))

    encoded_mime = "image/webp"
    try:
        encoded_bytes = _encode_webp(image)
    except OSError:
        # Fallback for runtime environments where WebP codec is unavailable.
        encoded_mime = "image/jpeg"
        encoded_bytes = _encode_jpeg(image)

    if len(encoded_bytes) > max_bytes:
        raise ImageValidationError(
            f"Processed image still too large ({len(encoded_bytes) / (1024 * 1024):.1f}MB). "
            f"Please lower MAX_IMAGE_SIDE or use a smaller image (target <= {max_upload_mb}MB)."
        )

    b64 = base64.b64encode(encoded_bytes).decode("ascii")
    image_hash = hashlib.sha256(encoded_bytes).hexdigest()

    return {
        "data_url": f"data:{encoded_mime};base64,{b64}",
        "hash": image_hash,
    }
