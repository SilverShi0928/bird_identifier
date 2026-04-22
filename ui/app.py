from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import load_settings
from core.classifier_service import ClassifierService
from core.deepseek_client import DeepSeekClient, DeepSeekClientError
from core.ebird_api_client import EbirdApiClient
from core.deepseek_translator import DeepSeekTranslator
from core.image_pipeline import ImageValidationError, prepare_image
from data.repository import HistoryRepository


def _append_debug_log(message: str) -> None:
    logs = st.session_state.setdefault("bird_identifier_debug_logs", [])
    logs.append(message)


def _zh_label_for_rank(predictions_zh: list[dict], rank: int) -> str:
    for p in predictions_zh or []:
        if int(p.get("rank", 0) or 0) == rank:
            return str(p.get("label_zh") or "").strip()
    return ""


def _top1_predictions_zh_entry(result: dict) -> dict | None:
    preds_zh = result.get("predictions_zh") or []
    top = next((p for p in preds_zh if int(p.get("rank", 0) or 0) == 1), None)
    if top is None and preds_zh:
        return preds_zh[0]
    return top


def _top1_features_zh_list(result: dict) -> list[str]:
    top = _top1_predictions_zh_entry(result)
    if not top:
        return []
    return [str(f).strip() for f in (top.get("features_zh") or []) if str(f).strip()]


def _top1_summary(result: dict) -> tuple[str, str, float]:
    """(primary_label_zh_or_en, secondary_en_name, confidence_or_rerank)."""
    preds = result.get("predictions") or []
    preds_zh = result.get("predictions_zh") or []
    if not preds:
        return "—", "", 0.0
    top = preds[0]
    conf = float(top.get("rerank_score", top.get("confidence", 0.0)))
    en = str(top.get("common_name") or top.get("species") or "").strip()
    zh = _zh_label_for_rank(preds_zh, 1)
    primary = zh or en or "—"
    return primary, en, conf


def _format_common_places_text(result: dict) -> str:
    places = result.get("hk_common_places_zh") or result.get("hk_common_places") or []
    if places:
        return "、".join(str(p) for p in places if p)
    ebird = result.get("ebird_recent") or {}
    if ebird.get("status") != "ok":
        return ""
    locs: list[str] = []
    for _region, rows in sorted((ebird.get("recent_by_region") or {}).items()):
        for r in rows or []:
            ln = r.get("locName")
            if ln and str(ln) not in locs:
                locs.append(str(ln))
    return "、".join(locs[:8]) if locs else ""


def _render_ebird_species_page_link(result: dict) -> None:
    ebird_recent = result.get("ebird_recent") or {}
    if ebird_recent.get("status") != "ok":
        return
    species_url = ebird_recent.get("species_page_url", "")
    if species_url:
        st.markdown(f"[eBird 物種頁]({species_url})")


def render_deep_analysis_expander(result: dict) -> None:
    translation_status = result.get("translation_status", "")
    if translation_status == "success":
        st.caption("翻譯：已套用繁體中文。")
    elif translation_status == "failed":
        st.caption("翻譯失敗，以下保留英文為主。")
    elif translation_status == "disabled":
        st.caption("未啟用翻譯（未設定翻譯用 API 金鑰）。")

    if not result.get("is_bird", True):
        st.warning("模型判斷呢張圖未必係雀鳥。")

    preds = result.get("predictions", [])
    preds_zh = result.get("predictions_zh") or []

    if preds:
        st.markdown("**模型候選（原文）**")
        for idx, pred in enumerate(preds, start=1):
            species_label = pred.get("common_name") or pred.get("species") or "Unknown"
            confidence = float(pred.get("confidence", 0.0))
            line = f"Top {idx}：{species_label}（{confidence:.0%}）"
            if "model_confidence" in pred:
                line += (
                    f" — 重排：模型 {float(pred.get('model_confidence', 0.0)):.0%}，"
                    f"地區 {float(pred.get('region_bonus', 0.0)):+.0%}，"
                    f"最終 {float(pred.get('rerank_score', confidence)):.0%}"
                )
            st.write(line)
            features = pred.get("features") or []
            if features:
                st.caption("關鍵特徵：" + "，".join(str(f) for f in features))

    if preds_zh:
        st.markdown("**中文候選（翻譯）**")
        for item in preds_zh:
            rank = item.get("rank", 0)
            label_zh = item.get("label_zh", "")
            st.write(f"Top {rank}：{label_zh}")
            features_zh = item.get("features_zh") or []
            if features_zh:
                st.caption("中文特徵：" + "、".join(str(f) for f in features_zh))

    if result.get("reasoning"):
        st.markdown("**說明**")
        st.write(result["reasoning"])

    hk_rarity = result.get("hk_rarity_note_zh") or result.get("hk_rarity_note") or ""
    hk_places = result.get("hk_common_places_zh") or result.get("hk_common_places") or []
    if hk_rarity or hk_places:
        st.markdown("**香港地理與罕見度**")
        if hk_rarity:
            st.write(f"- 罕見度：{hk_rarity}")
        if hk_places:
            st.write(f"- 常見地點：{'、'.join(str(p) for p in hk_places)}")

    if result.get("reasoning_zh"):
        st.markdown("**中文說明**")
        st.write(result["reasoning_zh"])

    ebird_recent = result.get("ebird_recent") or {}
    if not ebird_recent:
        return
    st.markdown("**eBird 最近觀測（API）**")
    status = ebird_recent.get("status", "")
    if status == "no_token":
        st.info("未設定 EBIRD_API_TOKEN，略過 eBird 最近觀測。")
    elif status == "no_match":
        st.warning("相片 Top 候選未能對應 eBird taxonomy。")
    elif status == "api_error":
        st.warning(f"eBird API 錯誤：{ebird_recent.get('error', '')}")
    elif status == "ok":
        if ebird_recent.get("source_candidate"):
            st.caption(f"Taxonomy 匹配自候選：{ebird_recent['source_candidate']}")
        if ebird_recent.get("species_code"):
            st.caption(
                f"eBird：`{ebird_recent.get('com_name', '')}` / *{ebird_recent.get('sci_name', '')}* "
                f"（`{ebird_recent['species_code']}`）"
            )
        recent_by_region = ebird_recent.get("recent_by_region") or {}
        summary_parts: list[str] = []
        empty_regions: list[str] = []
        for region in sorted(recent_by_region.keys()):
            rows = recent_by_region.get(region) or []
            if not rows:
                empty_regions.append(str(region))
                continue
            summary_parts.append(f"{region} {len(rows)} 筆")
        if summary_parts:
            st.caption("最近觀測筆數：" + "；".join(summary_parts))
        if empty_regions:
            st.caption("以下地區最近無回傳紀錄：" + "、".join(empty_regions))


def render_result_summary_column(result: dict) -> None:
    translation_status = result.get("translation_status", "")
    if translation_status == "failed":
        st.warning("翻譯失敗，右欄以英文／原文顯示為主。")

    preds = result.get("predictions", [])
    preds_zh = result.get("predictions_zh") or []

    if not result.get("is_bird", True):
        st.warning("模型判斷未必係雀鳥。")

    if not preds and not preds_zh:
        st.info("未有可用候選。")
        if result.get("reasoning"):
            st.write(result["reasoning"])
        if result.get("reasoning_zh"):
            st.write(result["reasoning_zh"])
        return

    st.success("### ✅ 識別完成")
    primary, en_name, conf = _top1_summary(result)
    delta = f"{conf:.0%} 信心度"
    st.metric(label="最有可能的品種", value=primary, delta=delta)
    if en_name and primary != en_name:
        st.caption(f"英文名：{en_name}")

    reasoning_zh = (result.get("reasoning_zh") or "").strip()
    feats_zh = _top1_features_zh_list(result)
    tstat = result.get("translation_status", "")

    if tstat == "disabled":
        st.caption("中文說明／中文特徵：未啟用翻譯；請於 `.env` 設定翻譯用 API 金鑰（見 `.env.example`）。")
    else:
        if reasoning_zh:
            st.markdown("**中文說明**")
            st.write(reasoning_zh)
        else:
            st.caption("中文說明：（暫無內容）")

        if feats_zh:
            st.markdown("**中文特徵**")
            st.markdown("\n".join(f"- {f}" for f in feats_zh))
        else:
            st.caption("中文特徵：（暫無內容）")

    _render_ebird_species_page_link(result)

    st.divider()
    st.subheader("常見地點")
    places_txt = _format_common_places_text(result)
    if places_txt:
        st.info(f"📍 {places_txt}")
    else:
        st.info("📍 暫無地點資料（可設定翻譯或 eBird token 以豐富結果）。")

    rest = preds[1:] if len(preds) > 1 else []
    if rest:
        st.write("其他可能：")
        for idx, pred in enumerate(rest, start=2):
            label = _zh_label_for_rank(preds_zh, idx) or pred.get("common_name") or pred.get("species") or f"Top {idx}"
            pconf = float(pred.get("rerank_score", pred.get("confidence", 0.0)))
            pconf = min(max(pconf, 0.0), 1.0)
            st.progress(pconf, text=f"{label}（{pconf:.0%}）")


def init_services():
    settings = load_settings()
    repo = HistoryRepository(str(ROOT / settings.db_path))
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
    ebird_api = EbirdApiClient(api_token=settings.ebird_api_token, timeout_seconds=settings.timeout_seconds)
    classifier = ClassifierService(
        client=client,
        top_k=settings.top_k,
        ebird_api=ebird_api,
        ebird_regions=settings.ebird_regions,
        ebird_recent_max=settings.ebird_recent_max_per_region,
    )
    translator = DeepSeekTranslator(
        api_key=settings.deepseek_translate_api_key,
        base_url=settings.deepseek_translate_base_url,
        model=settings.deepseek_translate_model,
        timeout_seconds=settings.timeout_seconds,
    )
    return settings, repo, classifier, translator


def main():
    st.set_page_config(page_title="雀鳥辨識", layout="wide")
    st.title("雀鳥辨識")
    st.caption("拖放圖片到指定區域，支援 jpg / png / webp。")
    st.session_state.setdefault("bird_identifier_debug_logs", [])

    settings, repo, classifier, translator = init_services()

    page = st.sidebar.radio("頁面", ["識別", "歷史", "設定"])

    if page == "設定":
        st.subheader("目前設定")
        st.write(
            {
                "辨識 API 金鑰已設定": bool(settings.api_key),
                "翻譯 API 金鑰已設定": bool(settings.deepseek_translate_api_key),
                "視覺模型": settings.model,
                "請求逾時（秒）": settings.timeout_seconds,
                "失敗重試次數": settings.retry_count,
                "Top-K": settings.top_k,
                "上載上限（MB）": settings.max_upload_mb,
                "影像最長邊（px）": settings.max_image_side,
                "歷史資料庫路徑": settings.db_path,
                "eBird API 已設定": bool(settings.ebird_api_token),
                "eBird 地區": settings.ebird_regions,
                "每區最近觀測上限": settings.ebird_recent_max_per_region,
                "MOSS TTS 已啟用": bool(settings.moss_tts_nano_home and settings.moss_tts_cli),
                "MOSS TTS 後端": settings.moss_tts_backend,
                "MOSS TTS 聲線": settings.moss_tts_voice,
                "MOSS 參考音檔": settings.moss_tts_prompt_speech or "（未設定；使用內建 preset）",
                "粵語 TTS 引擎": settings.cantonese_tts_engine,
                "粵語朗讀聲線代號": settings.edge_tts_voice,
            }
        )
        st.info("如要修改設定，請編輯專案內嘅 `.env`，然後重新啟動程式（範例見 `.env.example`）。")
        return

    if page == "歷史":
        st.subheader("歷史記錄")
        rows = repo.list_recent(limit=100)
        if not rows:
            st.info("未有歷史記錄。")
            return
        st.dataframe(rows, width="stretch")
        return

    st.subheader("拖放上載區")
    uploaded = st.file_uploader(
        "將圖片拖入呢個框，或點擊選檔",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
        help=f"最大 {settings.max_upload_mb}MB。拖放到此處即可上載。",
    )

    if uploaded is None:
        st.info("等待上載圖片...")
        return

    file_bytes = uploaded.getvalue()

    if st.button("開始識別", type="primary"):
        try:
            st.session_state["bird_identifier_cache_notice"] = ""
            st.session_state["bird_identifier_debug_logs"] = []
            _append_debug_log("[識別] 開始")
            with st.spinner("識別中..."):
                _append_debug_log("[識別] 準備圖片（驗證、縮圖、轉 data URL）")
                prepared = prepare_image(
                    file_bytes=file_bytes,
                    max_upload_mb=settings.max_upload_mb,
                    max_image_side=settings.max_image_side,
                )
                _append_debug_log(f"[識別] 圖片 prepared hash={prepared['hash']}")
                mode_key = "photo_with_ebird_recent"
                cache_key = f"{mode_key}:{prepared['hash']}"
                _append_debug_log(f"[識別] 模式={mode_key}，檢查快取 key={cache_key}")
                cached = repo.get_latest_success_by_hash(cache_key)
                if cached:
                    result = cached
                    st.session_state["bird_identifier_cache_notice"] = "命中快取：已使用歷史結果。"
                    _append_debug_log("[識別] 命中快取，直接使用歷史結果")
                else:
                    _append_debug_log("[識別] 呼叫 classify（相片 + eBird 最近觀測）")
                    result = classifier.classify(prepared["data_url"])
                    _append_debug_log("[識別] 拉取 eBird 最近觀測")
                    result["ebird_recent"] = classifier.fetch_ebird_recent_for_predictions(
                        result.get("predictions", [])
                    )
                    _append_debug_log("[識別] 依 eBird 結果重排候選")
                    result["predictions"] = classifier.rerank_predictions_with_ebird_recent(
                        result.get("predictions", []),
                        result.get("ebird_recent", {}),
                    )
                    if translator.enabled:
                        _append_debug_log("[識別] 呼叫翻譯 API")
                        translated = translator.translate_result(result)
                        result["reasoning_zh"] = translated.get("reasoning_zh", "")
                        result["predictions_zh"] = translated.get("predictions_zh", [])
                        result["hk_rarity_note_zh"] = translated.get("hk_rarity_note_zh", "")
                        result["hk_common_places_zh"] = translated.get("hk_common_places_zh", [])
                        has_translation = bool(result["reasoning_zh"] or result["predictions_zh"])
                        result["translation_status"] = "success" if has_translation else "failed"
                        _append_debug_log(f"[識別] 翻譯完成，status={result['translation_status']}")
                    else:
                        result["translation_status"] = "disabled"
                        _append_debug_log("[識別] 未啟用翻譯")
                    result["identify_mode"] = mode_key
                    repo.add_record(
                        image_hash=cache_key,
                        file_name=uploaded.name,
                        result=result,
                    )
                    _append_debug_log("[識別] 結果已寫入資料庫")
                st.session_state["bird_identifier_last_result"] = result
                st.session_state["bird_identifier_last_file"] = uploaded.name
            _append_debug_log("[識別] 完成")
        except (ImageValidationError, DeepSeekClientError, ValueError) as exc:
            repo.add_record(
                image_hash="",
                file_name=uploaded.name,
                result=None,
                error=str(exc),
            )
            st.error(f"識別失敗: {exc}")
            _append_debug_log(f"[識別] 失敗：{exc}")
        except Exception as exc:  # noqa: BLE001
            repo.add_record(
                image_hash="",
                file_name=uploaded.name,
                result=None,
                error=f"Unexpected error: {exc}",
            )
            st.error(f"Unexpected error: {exc}")
            _append_debug_log(f"[識別] Unexpected error：{exc}")

    latest_result = st.session_state.get("bird_identifier_last_result")
    last_file = st.session_state.get("bird_identifier_last_file") or ""
    show_result = bool(latest_result and last_file == uploaded.name)

    if show_result and last_file:
        st.caption(f"目前結果：{last_file}")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.image(file_bytes, caption=uploaded.name, use_container_width=True)
        if show_result and latest_result:
            with st.expander("🔍 詳盡分析"):
                render_deep_analysis_expander(latest_result)

    with col2:
        notice = st.session_state.get("bird_identifier_cache_notice", "")
        if notice:
            st.info(notice)
        if show_result and latest_result:
            render_result_summary_column(latest_result)
        elif latest_result and last_file and last_file != uploaded.name:
            st.warning("已更換上載檔案，請重新按「開始識別」。")
        else:
            st.info("👆 點擊「開始識別」以取得結果。")

    with st.expander("Debug 日誌"):
        logs = st.session_state.get("bird_identifier_debug_logs", [])
        if logs:
            st.code("\n".join(logs), language="text")
        else:
            st.caption("目前無 debug 日誌。")


if __name__ == "__main__":
    main()
