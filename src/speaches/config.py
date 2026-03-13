from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# 定义推理设备类型：cpu, cuda 或 auto（自动选择）
type Device = Literal["cpu", "cuda", "auto"]

# 定义量化类型，参考 CTranslate2 文档
# https://github.com/OpenNMT/CTranslate2/blob/master/docs/quantization.md#quantize-on-model-conversion
type Quantization = Literal[
    "int8", "int8_float16", "int8_bfloat16", "int8_float32", "int16", "float16", "bfloat16", "float32", "default"
]


class WhisperConfig(BaseModel):
    """Whisper 模型配置。
    详情请参考：https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py#L599
    """

    inference_device: Device = "auto"  # 推理设备
    device_index: int | list[int] = 0  # 设备索引，可以是单个整数或整数列表
    compute_type: Quantization = "default"  # 计算类型（量化方式）
    cpu_threads: int = 0  # CPU 线程数，0 表示自动
    num_workers: int = 1  # 工作进程数
    batch_size: int = 16  # 批处理大小


class OrtOptions(BaseModel):
    """ONNX Runtime (ORT) 选项配置。"""

    exclude_providers: list[str] = ["TensorrtExecutionProvider"]
    """
    要从推理会话中排除的 ORT 提供程序列表。
    """
    provider_priority: dict[str, int] = {"CUDAExecutionProvider": 100}
    """
    ORT 提供程序及其优先级的字典。值越大，优先级越高。
    如果未指定，提供程序的默认优先级为 0。
    """
    provider_opts: dict[str, dict[str, Any]] = {}
    """
    ORT 提供程序选项的字典。键是提供程序名称，值是选项字典。
    示例：{"CUDAExecutionProvider": {"cudnn_conv_algo_search": "DEFAULT"}}
    """


# TODO: 在文档字符串中记录 `alias` 的行为
class Config(BaseSettings):
    """应用程序配置类。值可以通过环境变量设置。

    Pydantic 会自动处理大写的环境变量到相应字段的映射。
    对于嵌套配置，环境变量应带有嵌套字段名和双下划线前缀。
    例如：
    环境变量 `LOG_LEVEL` 将映射到 `log_level`；
    `WHISPER__INFERENCE_DEVICE`（注意双下划线）将映射到 `whisper.inference_device`；
    要设置量化为 int8，使用 `WHISPER__COMPUTE_TYPE=int8` 等。
    """

    model_config = SettingsConfigDict(env_nested_delimiter="__")

    stt_model_ttl: int = Field(default=300, ge=-1)
    """
    语音转文本 (STT) 模型在最后一次使用后卸载前的等待时间（秒）。
    -1: 永不卸载模型。
    0: 使用后立即卸载模型。
    """

    tts_model_ttl: int = Field(default=300, ge=-1)
    """
    文本转语音 (TTS) 模型在最后一次使用后卸载前的等待时间（秒）。
    -1: 永不卸载模型。
    0: 使用后立即卸载模型。
    """

    vad_model_ttl: int = Field(default=-1, ge=-1)
    """
    语音活动检测 (VAD) 模型在最后一次使用后卸载前的等待时间（秒）。
    -1: 永不卸载模型。
    0: 使用后立即卸载模型。
    """

    api_key: SecretStr | None = None
    """
    如果设置，所有 API 请求都需要此 API 密钥。
    以下端点无需身份验证即可公开访问：
    - /health (健康检查端点)
    - /docs (API 文档)
    - /openapi.json (OpenAPI 架构)
    - Web UI (Gradio 界面)
    """
    log_level: str = "debug"
    """
    日志级别。可选：'debug', 'info', 'warning', 'error', 'critical'。
    """
    host: str = Field(alias="UVICORN_HOST", default="0.0.0.0")  # 服务监听地址
    port: int = Field(alias="UVICORN_PORT", default=8000)      # 服务监听端口
    allow_origins: list[str] | None = None
    """
    允许的跨域来源列表。参考 Pydantic Settings 文档。
    用法示例：
        `export ALLOW_ORIGINS='["http://localhost:3000", "http://localhost:3001"]'`
        `export ALLOW_ORIGINS='["*"]'`
    """

    enable_ui: bool = True
    """
    是否启用 Gradio UI。如果你想减少依赖并略微提高启动速度，可以禁用此项。
    """

    whisper: WhisperConfig = WhisperConfig()

    # TODO: 从字段名称中移除下划线前缀
    _unstable_vad_filter: bool = True
    """
    语音识别端点中 VAD (语音活动检测) 过滤器的默认值。
    启用后，模型将过滤掉非语音片段。有助于消除由于背景静音引起的语音识别幻觉。

    注意：设置 `_unstable_vad_filter: True` 在技术上偏离了 OpenAI API 规范，
    因此你可能希望将其设置为 `False`。

    注意：这是一个不稳定功能，未来可能会更改。
    """

    loopback_host_url: str | None = None
    """
    如果设置，这是 gradio 应用用于连接到托管 speaches 的 API 服务器的 URL。
    如果未设置，gradio 应用将使用用户连接到 gradio 应用时所使用的 URL。
    """

    # TODO: 记录以下配置选项
    chat_completion_base_url: str = "http://localhost:11434/v1"
    chat_completion_api_key: SecretStr = SecretStr("cant-be-empty")

    unstable_ort_opts: OrtOptions = OrtOptions()

    otel_exporter_otlp_endpoint: str | None = None
    """
    OpenTelemetry OTLP 导出器端点。如果设置，将启用遥测。
    示例：'http://localhost:4317'
    该设置会覆盖 OTEL_EXPORTER_OTLP_ENDPOINT 环境变量。
    """

    otel_service_name: str = "speaches"
    """
    用于在追踪中识别此应用程序的 OpenTelemetry 服务名称。
    该设置会覆盖 OTEL_SERVICE_NAME 环境变量。
    """

    preload_models: list[str] = []
    """
    在应用程序启动期间下载的模型 ID 列表。
    如果本地尚不存在，模型将按顺序下载。
    如果任何模型下载失败或在注册表中未找到，应用程序将退出。
    示例：["Systran/faster-whisper-tiny", "rhasspy/piper-voices"]
    """
