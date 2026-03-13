# Speaches 自定义镜像部署指南

## 概述

基于 speaches 官方代码，新增 `batch_size` 可配置项，通过环境变量适配不同 GPU 环境。

### 改动内容

| 文件 | 改动 |
|------|------|
| `src/speaches/config.py` | `WhisperConfig` 添加 `batch_size: int = 16` |
| `src/speaches/executors/whisper.py` | 3 处 `transcribe()` 调用传入 `batch_size` |

### 核心参数说明

| 环境变量 | 映射到 | 作用 | 建议值 |
|---------|--------|------|--------|
| `WHISPER__BATCH_SIZE` | `WhisperConfig.batch_size` | VAD 切分后同时送入 GPU 处理的 chunk 数量，**最关键的性能参数** | A4000: 8, H100: 64 |
| `WHISPER__COMPUTE_TYPE` | `WhisperConfig.compute_type` | 模型量化精度，`float16` 比默认 `float32` 显存减半且更快 | `float16` |
| `WHISPER__INFERENCE_DEVICE` | `WhisperConfig.inference_device` | 推理设备 | `cuda` |
| `WHISPER__DEVICE_INDEX` | `WhisperConfig.device_index` | 使用哪块 GPU（0-based） | `0` |
| `WHISPER__NUM_WORKERS` | `WhisperConfig.num_workers` | CTranslate2 的 `inter_threads`，控制并发请求处理数。单 GPU 单用户场景设 1 | `1` |
| `WHISPER__CPU_THREADS` | `WhisperConfig.cpu_threads` | CTranslate2 的 `intra_threads`，CPU 线程数。GPU 推理场景影响极小 | `0` (auto) |
| `STT_MODEL_TTL` | `Config.stt_model_ttl` | 模型空闲后卸载等待时间（秒），`-1` 为常驻 | `-1` |
| `VAD_MODEL_TTL` | `Config.vad_model_ttl` | VAD 模型空闲后卸载等待时间 | `-1` |
| `PRELOAD_MODELS` | `Config.preload_models` | 启动时预下载的模型列表 | 按需填写 |

> **环境变量命名规则**：嵌套字段用双下划线 `__` 分隔，如 `WHISPER__BATCH_SIZE` 对应 `config.whisper.batch_size`。由 pydantic-settings 的 `env_nested_delimiter="__"` 处理。

---

## 构建镜像

```bash
cd /home/ysnow/references/speaches

docker build -f Dockerfile -t speaches-custom:1.0.0 \
  --build-arg BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 \
  --build-arg http_proxy=http://host.docker.internal:10090 \
  --build-arg https_proxy=http://host.docker.internal:10090 \
  .
```

> 如果公司网络不需要代理，去掉 `--build-arg *_proxy` 参数。

导出镜像以便传到测试机：

```bash
docker save speaches-custom:1.0.0 | gzip > speaches-custom-1.0.0.tar.gz
# 传到测试机后：
docker load < speaches-custom-1.0.0.tar.gz
```

---

## 环境配置

### A4000 16GB（本机开发测试）

文件：`.env.a4000`

```env
# A4000 16GB (local dev/test)
WHISPER__BATCH_SIZE=8
WHISPER__COMPUTE_TYPE=float16
WHISPER__INFERENCE_DEVICE=cuda
WHISPER__DEVICE_INDEX=0
WHISPER__NUM_WORKERS=1

STT_MODEL_TTL=-1
VAD_MODEL_TTL=-1
```

显存预估（两个模型同时加载）：
- turbo 模型权重 (fp16): ~1.5 GB
- Belle 模型权重 (fp16): ~2.9 GB
- batch_size=8 活跃张量: ~2-3 GB
- 总计: ~7-8 GB（16 GB 安全）

### H100 80GB（测试机 demo）

文件：`.env.h100`

```env
# H100 80GB (demo server, single GPU)
WHISPER__BATCH_SIZE=64
WHISPER__COMPUTE_TYPE=float16
WHISPER__INFERENCE_DEVICE=cuda
WHISPER__DEVICE_INDEX=0
WHISPER__NUM_WORKERS=1

STT_MODEL_TTL=-1
VAD_MODEL_TTL=-1

PRELOAD_MODELS='["deepdml/faster-whisper-large-v3-turbo-ct2","k1nto/Belle-whisper-large-v3-zh-punct-ct2"]'
```

> 只用一块 H100 即可。`device_index=[0,1]` 双卡模式是模型复制（非拆分），对单用户 demo 无加速效果。

---

## 部署方式

### 方式一：docker run

```bash
# A4000 本机
docker run --gpus all -d --name speaches -p 8102:8000 \
  -v ~/models/audio/huggingface/hub:/home/ubuntu/.cache/huggingface/hub \
  --env-file .env.a4000 \
  speaches-custom:1.0.0

# H100 测试机
docker run --gpus '"device=0"' -d --name speaches -p 8102:8000 \
  -v /path/to/models:/home/ubuntu/.cache/huggingface/hub \
  --env-file .env.h100 \
  speaches-custom:1.0.0
```

### 方式二：docker-compose（推荐）

文件：`docker-compose.yml`

```yaml
services:
  speaches:
    image: speaches-custom:1.0.0
    container_name: speaches
    restart: unless-stopped
    ports:
      - "8102:8000"
    volumes:
      - ${MODEL_CACHE_DIR:-~/models/audio/huggingface/hub}:/home/ubuntu/.cache/huggingface/hub
    env_file:
      - .env
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

A4000 本机启动：

```bash
cp .env.a4000 .env
docker compose up -d
```

H100 测试机启动：

```bash
cp .env.h100 .env
MODEL_CACHE_DIR=/path/to/models docker compose up -d
```

---

## A/B 测试方案（测试机）

在测试机上同时运行原版和新版容器，用不同端口区分，对同一段音频做对比。

### docker-compose.ab-test.yml

```yaml
services:
  speaches-original:
    image: ghcr.io/speaches-ai/speaches:latest-cuda
    container_name: speaches-original
    ports:
      - "8102:8000"
    volumes:
      - ${MODEL_CACHE_DIR:-~/models/audio/huggingface/hub}:/home/ubuntu/.cache/huggingface/hub
    environment:
      - WHISPER__COMPUTE_TYPE=float16
      - WHISPER__INFERENCE_DEVICE=cuda
      - STT_MODEL_TTL=-1
      - VAD_MODEL_TTL=-1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]
              capabilities: [gpu]

  speaches-custom:
    image: speaches-custom:1.0.0
    container_name: speaches-custom
    ports:
      - "8103:8000"
    volumes:
      - ${MODEL_CACHE_DIR:-~/models/audio/huggingface/hub}:/home/ubuntu/.cache/huggingface/hub
    environment:
      - WHISPER__BATCH_SIZE=64
      - WHISPER__COMPUTE_TYPE=float16
      - WHISPER__INFERENCE_DEVICE=cuda
      - WHISPER__DEVICE_INDEX=0
      - WHISPER__NUM_WORKERS=1
      - STT_MODEL_TTL=-1
      - VAD_MODEL_TTL=-1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["1"]
              capabilities: [gpu]
```

> 关键：原版绑 GPU 0，新版绑 GPU 1，互不干扰。

### 启动 A/B 测试

```bash
docker compose -f docker-compose.ab-test.yml up -d

# 等待两个容器就绪
sleep 10
curl -s http://localhost:8102/health
curl -s http://localhost:8103/health
```

### 测试脚本

准备一段 2 分钟以上的真实中文音频（如会议录音），命名为 `test_audio.wav`。

```bash
#!/bin/bash
AUDIO="test_audio.wav"
MODELS=("deepdml/faster-whisper-large-v3-turbo-ct2" "k1nto/Belle-whisper-large-v3-zh-punct-ct2")
MODEL_LABELS=("turbo" "Belle")
PORTS=("8102" "8103")
PORT_LABELS=("original(bs=16)" "custom(bs=64)")
RUNS=3

echo "============================================================"
echo " Speaches A/B Benchmark (H100)"
echo " Audio: $(ffprobe -v quiet -show_entries format=duration \
       -of csv=p=0 $AUDIO 2>/dev/null | awk '{printf "%.0fs", $1}')"
echo " Runs per config: $RUNS"
echo "============================================================"

for mi in 0 1; do
  model="${MODELS[$mi]}"
  mlabel="${MODEL_LABELS[$mi]}"

  echo ""
  echo "--- Model: $mlabel ---"
  printf "%-20s | %8s | %8s | %8s | %8s\n" "Container" "Run1" "Run2" "Run3" "Avg"
  printf "%-20s-+-%8s-+-%8s-+-%8s-+-%8s\n" \
    "--------------------" "--------" "--------" "--------" "--------"

  for pi in 0 1; do
    port="${PORTS[$pi]}"
    plabel="${PORT_LABELS[$pi]}"

    # Warmup
    curl -s -X POST "http://localhost:${port}/v1/audio/transcriptions" \
      -F "file=@${AUDIO}" -F "model=${model}" > /dev/null 2>&1

    times=()
    total=0
    for r in $(seq 1 $RUNS); do
      secs=$( { time curl -s -X POST \
        "http://localhost:${port}/v1/audio/transcriptions" \
        -F "file=@${AUDIO}" -F "model=${model}" \
        > /dev/null 2>&1; } 2>&1 \
        | grep real \
        | awk '{split($2,a,"m"); split(a[2],b,"s"); print a[1]*60+b[1]}')
      times+=("$secs")
      total=$(echo "$total + $secs" | bc)
    done

    avg=$(echo "scale=3; $total / $RUNS" | bc)
    printf "%-20s | %8ss | %8ss | %8ss | %8ss\n" \
      "$plabel" "${times[0]}" "${times[1]}" "${times[2]}" "$avg"
  done
done

echo ""
echo "--- GPU Memory Usage ---"
nvidia-smi --query-gpu=index,name,memory.used,memory.total \
  --format=csv,noheader
```

### 清理

```bash
docker compose -f docker-compose.ab-test.yml down
```

---

## A4000 实测数据（参考）

测试条件：A4000 16GB，2 分钟真实中文会议音频，独占 GPU，3 次取平均。

| 模型 | 原版 (bs=16) | 新版 (bs=8) | 提速 |
|------|-------------|------------|------|
| turbo | 6.20s | 5.90s | 5% |
| Belle (large-v3) | 13.59s | 7.28s | **46%** |

> A4000 上 `batch_size=8` 优于默认 16，因为 16GB 显存下 bs=16 触发了显存压力。
> H100 80GB 显存充裕，`batch_size=64` 预期表现更好。

---

## 常见问题

### batch_size 设多大合适？

取决于 GPU 显存。经验值：

| GPU | 显存 | 推荐 batch_size |
|-----|------|----------------|
| A4000 | 16 GB | 8 |
| RTX 4090 | 24 GB | 16 |
| A100 | 40/80 GB | 32-64 |
| H100 | 80 GB | 64 |

如果 OOM，减半重试。

### num_workers 要不要调大？

单 GPU 单用户场景保持 `1`。实测在 A4000 单卡上 `num_workers` 越大并发越慢（GPU 争抢）。
多 GPU + 高并发场景才考虑调大。

### device_index=[0,1] 双卡有用吗？

CTranslate2 的双卡是**模型复制**（每张卡加载一份完整模型），不是模型拆分。
单用户场景无加速效果，仅在多并发时通过分配到不同 GPU 提升吞吐。
Demo 场景直接用单卡 + 大 batch_size 即可。
