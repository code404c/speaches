# Speaches (Custom Fork)

Fork of [speaches-ai/speaches](https://github.com/speaches-ai/speaches) with custom configuration for internal deployment.

> Original project README: [README.upstream.md](./README.upstream.md)

## Changes from Upstream

- **`batch_size` configuration**: Added `batch_size` field to `WhisperConfig` (default: 16), controllable via `WHISPER__BATCH_SIZE` environment variable
- **Docker Compose**: Custom `docker-compose.yml` with pre-built image `speaches-custom:1.0.0`, exposed on port `8102`
- **Environment presets**: `.env.h100` for H100 GPU deployment

## Quick Start

```bash
# Start the service
docker compose up -d

# Check logs
docker compose logs -f speaches
```

## Configuration

Environment variables are configured via `.env` file (create from a preset):

```bash
cp .env.h100 .env   # H100 preset
# Edit .env as needed
```

Key environment variables:

| Variable | Description | Default |
|---|---|---|
| `WHISPER__BATCH_SIZE` | Batch size for Whisper inference | `16` |
| `WHISPER__COMPUTE_TYPE` | Quantization type (`float16`, `int8`, etc.) | `default` |
| `WHISPER__INFERENCE_DEVICE` | Device (`cpu`, `cuda`, `auto`) | `auto` |
| `STT_MODEL_TTL` | STT model unload timeout in seconds (`-1` = never) | `300` |
| `PRELOAD_MODELS` | JSON list of models to preload at startup | `[]` |
| `MODEL_CACHE_DIR` | Host path for model cache (docker-compose volume) | `~/models/audio/huggingface/hub` |

## Docker Image

The custom image `speaches-custom:1.0.0` is pre-built. To rebuild:

```bash
docker build -t speaches-custom:1.0.0 .
```

## Project Structure

```
docker-compose.yml     # Custom deployment config
.env.h100              # Environment preset for H100
src/speaches/config.py # Application config (with batch_size addition)
README.upstream.md     # Original upstream README
```
