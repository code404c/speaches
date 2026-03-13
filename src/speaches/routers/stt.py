import asyncio
from collections.abc import Generator
import logging
from typing import Annotated, Literal

from fastapi import (
    APIRouter,
    Form,
    Request,
    Response,
)
from fastapi.responses import StreamingResponse
import openai.types.audio

from speaches.api_types import (
    DEFAULT_TIMESTAMP_GRANULARITIES,
    TIMESTAMP_GRANULARITIES_COMBINATIONS,
    TimestampGranularities,
)
from speaches.dependencies import (
    AudioFileDependency,
    ExecutorRegistryDependency,
)
from speaches.executors.shared.handler_protocol import (
    NonStreamingTranscriptionResponse,
    StreamingTranscriptionEvent,
    TranscriptionRequest,
    TranslationRequest,
    TranslationResponse,
    VadRequest,
)
from speaches.executors.silero_vad_v5 import VadOptions
from speaches.model_aliases import ModelId
from speaches.routers.utils import find_executor_for_model_or_raise, get_model_card_data_or_raise
from speaches.text_utils import format_as_sse

logger = logging.getLogger(__name__)

# 定义 STT 路由，标签为自动语音识别 (ASR)
router = APIRouter(tags=["automatic-speech-recognition"])

# 支持的响应格式类型
type ResponseFormat = Literal["text", "json", "verbose_json", "srt", "vtt"]
RESPONSE_FORMATS = ("text", "json", "verbose_json", "srt", "vtt")

# 默认响应格式为 json，符合 OpenAI API 规范
# https://platform.openai.com/docs/api-reference/audio/createTranscription#audio-createtranscription-response_format
DEFAULT_RESPONSE_FORMAT: ResponseFormat = "json"

# 默认的 VAD (语音活动检测) 选项，参考 faster_whisper
DEFAULT_VAD_OPTIONS = VadOptions(min_silence_duration_ms=160, max_speech_duration_s=30)


def translation_response_to_http_response(res: TranslationResponse) -> Response:  # noqa: RET503
    """将翻译响应转换为 HTTP 响应对象。"""
    if isinstance(res, tuple):
        # 处理原始文本响应（如 SRT/VTT）
        text, media_type = res
        return Response(content=text, media_type=media_type)
    elif isinstance(res, (openai.types.audio.Translation, openai.types.audio.TranslationVerbose)):
        # 处理 JSON 响应
        return Response(content=res.model_dump_json(), media_type="application/json")


@router.post(
    "/v1/audio/translations",
    response_model=str | openai.types.audio.Translation | openai.types.audio.TranslationVerbose,
)
def translate_file(
    executor_registry: ExecutorRegistryDependency,
    audio: AudioFileDependency,
    model: Annotated[ModelId, Form()],
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[ResponseFormat, Form()] = DEFAULT_RESPONSE_FORMAT,
    temperature: Annotated[float, Form()] = 0.0,
) -> Response:
    """音频翻译接口：将语音翻译为英文。"""
    model_card_data = get_model_card_data_or_raise(model)
    # 查找负责翻译任务的执行器
    executor = find_executor_for_model_or_raise(model, model_card_data, executor_registry.translation)

    # 首先执行 VAD，识别有效语音片段，减少静音引起的幻觉
    vad_request = VadRequest(audio=audio, vad_options=DEFAULT_VAD_OPTIONS)
    speech_segments = executor_registry.vad.model_manager.handle_vad_request(vad_request)

    # 构造翻译请求并调用执行器
    translation_request = TranslationRequest(
        audio=audio,
        model=model,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        speech_segments=speech_segments,
        vad_options=DEFAULT_VAD_OPTIONS,
    )
    res = executor.model_manager.handle_translation_request(translation_request)
    return translation_response_to_http_response(res)


async def get_timestamp_granularities(request: Request) -> TimestampGranularities:
    """
    由于 Form() 不支持带有中括号的别名（如 timestamp_granularities[]），
    我们需要手动从请求表单中解析此参数。
    """
    form = await request.form()
    if form.get("timestamp_granularities[]") is None:
        return DEFAULT_TIMESTAMP_GRANULARITIES
    timestamp_granularities = form.getlist("timestamp_granularities[]")
    assert timestamp_granularities in TIMESTAMP_GRANULARITIES_COMBINATIONS, (
        f"{timestamp_granularities} 不是有效的 `timestamp_granularities[]` 值。"
    )
    return timestamp_granularities  # pyright: ignore[reportReturnType]


def transcription_response_to_http_response(
    res: NonStreamingTranscriptionResponse | Generator[StreamingTranscriptionEvent],
) -> Response | StreamingResponse:
    """将转录响应转换为 HTTP 响应或流式响应。"""
    if isinstance(res, tuple):
        # 原始文本响应
        text, media_type = res
        return Response(content=text, media_type=media_type)
    elif isinstance(res, (openai.types.audio.Transcription, openai.types.audio.TranscriptionVerbose)):
        # 非流式 JSON 响应
        return Response(content=res.model_dump_json(), media_type="application/json")
    else:
        # 流式响应：使用 SSE (Server-Sent Events) 格式
        return StreamingResponse(
            (format_as_sse(x.model_dump_json()) for x in res),
            media_type="text/event-stream",
        )


@router.post(
    "/v1/audio/transcriptions",
    response_model=str | openai.types.audio.Transcription | openai.types.audio.TranscriptionVerbose,
)
def transcribe_file(
    executor_registry: ExecutorRegistryDependency,
    request: Request,
    audio: AudioFileDependency,
    model: Annotated[ModelId, Form()],
    language: Annotated[str | None, Form()] = None,
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[ResponseFormat, Form()] = DEFAULT_RESPONSE_FORMAT,
    temperature: Annotated[float, Form()] = 0.0,
    timestamp_granularities: Annotated[
        TimestampGranularities,
        # 注意：这里的 alias 实际上在 multipart/form-data 中失效，需手动解析
        Form(alias="timestamp_granularities[]"),
    ] = ["segment"],
    stream: Annotated[bool, Form()] = False,
    # 非标准扩展参数
    hotwords: Annotated[str | None, Form()] = None,
    without_timestamps: Annotated[bool, Form()] = True,
) -> Response | StreamingResponse:
    """音频转录接口：将语音转换为文本。"""
    # 手动解析时间戳颗粒度选项
    timestamp_granularities = asyncio.run(get_timestamp_granularities(request))
    if timestamp_granularities != DEFAULT_TIMESTAMP_GRANULARITIES and response_format != "verbose_json":
        logger.warning(
            "仅当 `response_format` 为 `verbose_json` 时，提供 `timestamp_granularities[]` 才有意义。"
        )

    transcription_model_card_data = get_model_card_data_or_raise(model)
    # 查找负责转录任务的执行器
    transcription_executor = find_executor_for_model_or_raise(
        model, transcription_model_card_data, executor_registry.transcription
    )

    # 执行 VAD
    vad_request = VadRequest(audio=audio, vad_options=DEFAULT_VAD_OPTIONS)
    speech_segments = executor_registry.vad.model_manager.handle_vad_request(vad_request)

    # 构造转录请求
    transcription_request = TranscriptionRequest(
        audio=audio,
        model=model,
        language=language,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        timestamp_granularities=timestamp_granularities,
        stream=stream,
        hotwords=hotwords,
        speech_segments=speech_segments,
        vad_options=DEFAULT_VAD_OPTIONS,
        without_timestamps=without_timestamps,
    )
    # 调用执行器并返回响应
    res = transcription_executor.model_manager.handle_transcription_request(transcription_request)
    http_res = transcription_response_to_http_response(res)
    return http_res
