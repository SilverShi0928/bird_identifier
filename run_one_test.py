from pathlib import Path
import sys

from config.settings import load_settings
from core.classifier_service import ClassifierService
from core.deepseek_client import DeepSeekClient
from core.deepseek_translator import DeepSeekTranslator
from core.image_pipeline import prepare_image


def main() -> None:
    image_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(
            r"C:\Users\Silver.Shi\.cursor\projects\d-Python-Cursor\assets\c__Users_Silver.Shi_AppData_Roaming_Cursor_User_workspaceStorage_2841bc2952259eb5e4e63c3b0487ce9a_images_3D4A2398-__-1c04e0d1-9c87-4ee1-b062-f56713bd6791.png"
        )
    )
    settings = load_settings()

    prepared = prepare_image(
        file_bytes=image_path.read_bytes(),
        max_upload_mb=settings.max_upload_mb,
        max_image_side=settings.max_image_side,
    )
    client = DeepSeekClient(
        base_url=settings.base_url,
        api_key=settings.api_key,
        model=settings.model,
        timeout_seconds=settings.timeout_seconds,
        retry_count=settings.retry_count,
        openrouter_provider_ignore=settings.openrouter_provider_ignore,
        openrouter_http_referer=settings.openrouter_http_referer,
        openrouter_title=settings.openrouter_title,
    )
    service = ClassifierService(client=client, top_k=settings.top_k)
    translator = DeepSeekTranslator(
        api_key=settings.deepseek_translate_api_key,
        base_url=settings.deepseek_translate_base_url,
        model=settings.deepseek_translate_model,
        timeout_seconds=settings.timeout_seconds,
    )
    try:
        result = service.classify(prepared["data_url"])
    except Exception as exc:  # noqa: BLE001
        raw = client.classify_bird(prepared["data_url"], "Identify this bird and return JSON only.")
        print("classification_error =", exc)
        print("raw_response =", raw[:1200])
        return

    print("is_bird =", result.get("is_bird"))
    print("reasoning =", str(result.get("reasoning", ""))[:400])
    predictions = result.get("predictions", [])
    print("pred_count =", len(predictions))
    for index, pred in enumerate(predictions, start=1):
        label = pred.get("common_name") or pred.get("species") or "Unknown"
        confidence = float(pred.get("confidence", 0.0))
        features = ", ".join(pred.get("features", []))
        print(f"{index}. {label} ({confidence:.2%}) | features: {features}")

    if translator.enabled:
        translated = translator.translate_result(result)
        print("translation_predictions =", len(translated.get("predictions_zh", [])))
        print("reasoning_zh =", str(translated.get("reasoning_zh", ""))[:400])
        for item in translated.get("predictions_zh", []):
            rank = item.get("rank", 0)
            label_zh = item.get("label_zh", "")
            features_zh = "、".join(item.get("features_zh", []))
            print(f"ZH Top {rank}. {label_zh} | 中文特徵: {features_zh}")


if __name__ == "__main__":
    main()
