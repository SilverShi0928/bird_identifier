from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from core.moss_tts_service import MossTtsConfig, MossTtsError, MossTtsService


def test_build_top1_cantonese_text_uses_rank1():
    service = MossTtsService(MossTtsConfig(nano_home="x", cli="moss-tts-nano"))
    text = service.build_top1_cantonese_text(
        {
            "predictions_zh": [
                {"rank": 2, "label_zh": "第二名", "features_zh": ["特徵B"]},
                {"rank": 1, "label_zh": "普通翠鳥", "features_zh": ["亮藍綠色羽毛", "細長尖銳的鳥喙"]},
            ]
        }
    )
    assert "第一名係普通翠鳥" in text
    assert "亮藍綠色羽毛、細長尖銳的鳥喙" in text


def test_build_top1_cantonese_text_requires_top1():
    service = MossTtsService(MossTtsConfig(nano_home="x", cli="moss-tts-nano"))
    with pytest.raises(MossTtsError):
        service.build_top1_cantonese_text({"predictions_zh": []})


def test_synthesize_top1_builds_command_and_returns_audio(tmp_path: Path):
    nano_home = tmp_path / "nano"
    nano_home.mkdir()
    output_wav = tmp_path / "out.wav"
    output_wav.write_bytes(b"RIFF")
    service = MossTtsService(
        MossTtsConfig(
            nano_home=str(nano_home),
            cli="moss-tts-nano",
            backend="onnx",
            voice="Junhao",
            timeout_seconds=10,
        )
    )
    result = {"predictions_zh": [{"rank": 1, "label_zh": "戴菊", "features_zh": ["體型小巧"]}]}

    def fake_run(cmd, cwd, capture_output, text, timeout, check):  # noqa: ARG001
        assert cwd == str(nano_home)
        assert cmd[0] == "moss-tts-nano"
        assert "--voice" in cmd
        out_idx = cmd.index("--output") + 1
        Path(cmd[out_idx]).write_bytes(output_wav.read_bytes())
        return subprocess.CompletedProcess(cmd, 0, "", "")

    with patch("core.moss_tts_service.subprocess.run", side_effect=fake_run):
        got = service.synthesize_top1(result)

    assert got.startswith(b"RIFF")
