# Bird Identifier

Desktop-style bird image identifier with a drag-and-drop UI (Streamlit), **OpenAI-compatible** vision APIs (`chat/completions` + `image_url`), and local history (SQLite).

## Features

- Drag-and-drop image upload area (jpg/jpeg/png/webp).
- Bird classification via `chat/completions` with structured JSON output.
- Top-K predictions with confidence and key features.
- Optional Traditional Chinese translation via a separate text `chat/completions` endpoint.
- Optional MOSS-TTS-Nano readout for Chinese Top 1 + features.
- Local history page with previous runs and errors.
- Hash-based cache: same image can reuse latest successful result.
- Works on Windows and macOS.

## Project Structure

- `ui/app.py`: Streamlit UI.
- `core/deepseek_client.py`: OpenAI-compatible API client (retry/timeout).
- `core/image_pipeline.py`: image validation and conversion.
- `core/classifier_service.py`: prompt + JSON parsing + top-k cleanup.
- `data/repository.py`: SQLite history storage.
- `config/settings.py`: `.env` settings loader.

## Setup

1. Go to project directory:
   - `cd tools/bird_identifier`
2. Obtain API keys from whatever **vision-capable** Chat Completions host you use (must accept `image_url` in messages).
3. Create `.env` from template if needed:
   - Windows: `copy .env.example .env`
   - macOS: `cp .env.example .env`
4. Edit `.env` — start from `.env.example`. Vision API key + model id are required; translation keys are optional. The loader accepts the neutral `BIRD_*` names in the template, and still understands older alternate key names (see `config/settings.py` for precedence).

## Run

- Windows: double-click `run_windows.bat`
- macOS:
  1. `chmod +x run_macos.command`
  2. double-click `run_macos.command` or run `./run_macos.command`

Manual run:

```bash
pip install -r requirements.txt
streamlit run ui/app.py
```

## Test

```bash
pytest -q
```

## Batch scan (no labels)

Put images in `Testing_Photo/` (or any folder), then:

```bash
python scripts/batch_scan.py Testing_Photo --out Testing_Photo/results.csv
```

CSV columns: `file`, `top1`–`top3` names, confidences, `error`. Rows with empty tops usually mean the vision model returned free text instead of structured JSON (same as in the UI).

## Notes

- Default DB file is `bird_identifier.db` in project root.
- You can change model/base URL/timeouts in `.env`.
- If API returns non-JSON text, app attempts JSON extraction fallback.
- **Accuracy**: smaller vision models are weaker on fine-grained species ID. Prefer a stronger vision model id from your provider’s catalog. The app also tries to recover candidate names from prose when the model ignores JSON.
- **Region / availability**: some model ids may return HTTP 403 depending on region or account entitlements. Pick a vision model your dashboard shows as available to you.
