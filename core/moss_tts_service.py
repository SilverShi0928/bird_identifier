from __future__ import annotations

import shlex
import subprocess
import tempfile
import time
import asyncio
from dataclasses import dataclass
from pathlib import Path


class MossTtsError(RuntimeError):
    """Raised when MOSS-TTS-Nano execution fails."""


def _split_command(value: str) -> list[str]:
    parts = shlex.split(value.strip(), posix=False)
    return parts if parts else ["moss-tts-nano"]


@dataclass(frozen=True)
class MossTtsConfig:
    nano_home: str
    cli: str
    backend: str = "onnx"
    voice: str = "Junhao"
    prompt_speech: str = ""
    timeout_seconds: int = 120
    engine: str = "moss"  # edge | moss
    edge_voice: str = "yue-HK-HiuGaaiNeural"


class MossTtsService:
    def __init__(self, config: MossTtsConfig):
        self._cfg = config

    @property
    def enabled(self) -> bool:
        if self._cfg.engine == "edge":
            return True
        return bool(self._cfg.nano_home and self._cfg.cli)

    def build_top1_cantonese_text(self, result: dict) -> str:
        preds_zh = result.get("predictions_zh") or []
        top = next((x for x in preds_zh if int(x.get("rank", 0)) == 1), preds_zh[0] if preds_zh else None)
        if not top:
            raise MossTtsError("未有中文 Top 1 候選，無法朗讀。")

        label = str(top.get("label_zh") or "").strip()
        if not label:
            raise MossTtsError("Top 1 中文名稱為空白，無法朗讀。")

        features = [str(x).strip() for x in (top.get("features_zh") or []) if str(x).strip()]
        features_text = "、".join(features) if features else "未提供"
        return f"中文候選，第一名係{label}。中文特徵：{features_text}。"

    def synthesize_top1(self, result: dict) -> bytes:
        audio, _ = self.synthesize_top1_with_debug(result)
        return audio

    def synthesize_top1_with_debug(self, result: dict) -> tuple[bytes, list[str]]:
        debug_logs: list[str] = []

        def _log(msg: str) -> None:
            debug_logs.append(msg)

        if not self.enabled:
            raise MossTtsError("未設定 MOSS TTS（請設定 MOSS_TTS_NANO_HOME 及 MOSS_TTS_CLI）。")
        _log(f"TTS 設定檢查：enabled=true engine={self._cfg.engine}")

        text = self.build_top1_cantonese_text(result)
        _log(f"TTS 朗讀文字已建立，長度={len(text)}")

        if self._cfg.engine == "edge":
            try:
                import edge_tts  # type: ignore
            except Exception as exc:  # noqa: BLE001
                raise MossTtsError("未安裝 edge-tts，請先 `pip install edge-tts`。") from exc
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                out_path = Path(tmp.name)
            _log(f"Edge TTS 輸出檔：{out_path}")
            _log(f"Edge TTS voice={self._cfg.edge_voice}")
            started = time.perf_counter()
            try:
                communicate = edge_tts.Communicate(text=text, voice=self._cfg.edge_voice)
                asyncio.run(communicate.save(str(out_path)))
            except Exception as exc:  # noqa: BLE001
                if self._cfg.nano_home and self._cfg.cli:
                    _log(f"Edge TTS 失敗，改用 MOSS fallback：{exc}")
                else:
                    raise MossTtsError(f"Edge TTS 生成失敗：{exc}") from exc
            else:
                elapsed = time.perf_counter() - started
                _log(f"Edge TTS 完成，耗時={elapsed:.2f}s")
                data = out_path.read_bytes()
                _log(f"Edge TTS 音檔讀取成功：{len(data)} bytes")
                return data, debug_logs

        nano_home = Path(self._cfg.nano_home).expanduser()
        if not nano_home.is_dir():
            raise MossTtsError(f"MOSS_TTS_NANO_HOME 無效：{nano_home}")
        _log(f"TTS 工作目錄：{nano_home}")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = Path(tmp.name)
        _log(f"TTS 臨時輸出檔：{out_path}")

        cmd = [
            *_split_command(self._cfg.cli),
            "generate",
            "--backend",
            self._cfg.backend,
            "--text",
            text,
            "--output",
            str(out_path),
        ]
        prompt = self._cfg.prompt_speech.strip()
        if prompt:
            cmd.extend(["--prompt-speech", prompt])
            _log("TTS 模式：voice clone（使用 prompt_speech）")
        elif self._cfg.backend == "onnx":
            cmd.extend(["--voice", self._cfg.voice])
            _log(f"TTS 模式：ONNX preset voice={self._cfg.voice}")

        _log("TTS 指令：" + " ".join(cmd))
        started = time.perf_counter()

        proc = subprocess.run(
            cmd,
            cwd=str(nano_home),
            capture_output=True,
            text=True,
            timeout=self._cfg.timeout_seconds,
            check=False,
        )
        elapsed = time.perf_counter() - started
        _log(f"TTS subprocess 完成：exit={proc.returncode}，耗時={elapsed:.2f}s")
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            _log("TTS 錯誤輸出：" + detail)
            raise MossTtsError(f"MOSS 生成失敗（exit {proc.returncode}）：{detail}")
        if not out_path.exists():
            raise MossTtsError("MOSS 已返回成功，但找不到輸出 WAV。")
        data = out_path.read_bytes()
        _log(f"TTS 音檔讀取成功：{len(data)} bytes")
        return data, debug_logs
