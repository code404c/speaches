from functools import lru_cache
import logging
from pathlib import Path
import time
from typing import Annotated, cast

import av.error
from fastapi import (
    Depends,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from faster_whisper.audio import decode_audio
from httpx import ASGITransport, AsyncClient
import numpy as np
from numpy import float32
from openai import AsyncOpenAI
from openai.resources.audio import AsyncSpeech, AsyncTranscriptions
from openai.resources.chat.completions import AsyncCompletions

from speaches.audio import Audio
from speaches.config import Config
from speaches.executors.shared.registry import ExecutorRegistry

logger = logging.getLogger(__name__)

# 注意：`get_config` 被直接调用而非仅作为子依赖，以便这些函数可以在 `FastAPI` 之外使用。

# 使用 lru_cache 确保配置仅实例化一次
# 参考：https://fastapi.tiangolo.com/advanced/settings/?h=setti#creating-the-settings-only-once-with-lru_cache
@lru_cache
def get_config() -> Config:
    """获取应用程序配置单例。"""
    return Config()


async def get_config_async() -> Config:
    """异步获取配置，供 FastAPI 依赖注入使用。"""
    return get_config()


# FastAPI 依赖项：配置
ConfigDependency = Annotated[Config, Depends(get_config_async)]


@lru_cache
def get_executor_registry() -> ExecutorRegistry:
    """获取模型执行器注册表单例。"""
    config = get_config()
    return ExecutorRegistry(config)


async def get_executor_registry_async() -> ExecutorRegistry:
    """异步获取执行器注册表。"""
    return get_executor_registry()


# FastAPI 依赖项：模型执行器注册表
ExecutorRegistryDependency = Annotated[ExecutorRegistry, Depends(get_executor_registry_async)]


# 定义 HTTP Bearer 认证方案
security = HTTPBearer(auto_error=False)


async def verify_api_key(
    config: ConfigDependency, credentials: HTTPAuthorizationCredentials | None = Depends(security)
) -> None:
    """验证请求中的 API 密钥。"""
    assert config.api_key is not None
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="需要 API 密钥。请使用 Authorization 请求头并采用 Bearer 方案提供 API 密钥。",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if credentials.credentials != config.api_key.get_secret_value():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无效的 API 密钥。提供的 API 密钥不正确。",
            headers={"WWW-Authenticate": "Bearer"},
        )


# FastAPI 身份验证依赖项
ApiKeyDependency = Depends(verify_api_key)


def audio_file_dependency(
    file: Annotated[UploadFile, Form()],
) -> Audio:
    """
    音频文件处理依赖项。
    将上传的文件解码为 16kHz 单声道的浮点型 NumPy 数组。
    支持常见的音频格式以及 PCM 原始音频。
    """
    try:
        logger.debug(
            f"正在解码音频文件: {file.filename}, 类型: {file.content_type}, 大小: {file.size}"
        )
        start = time.perf_counter()

        # 处理 PCM 原始音频
        if file.content_type in ("audio/pcm", "audio/raw"):
            logger.debug(f"检测到 {file.content_type}，作为 s16le 单声道解析")
            raw_bytes = file.file.read()
            audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
            # 归一化到 [-1, 1] 范围
            audio_data = cast("np.typing.NDArray[float32]", audio_int16.astype(np.float32) / 32768.0)
        else:
            # 使用 faster_whisper 的音频解码工具，采样率固定为 16000
            audio_data = cast("np.typing.NDArray[float32]", decode_audio(file.file, sampling_rate=16000))

        elapsed = time.perf_counter() - start
        audio = Audio(audio_data, sample_rate=16000, name=Path(file.filename).stem if file.filename else None)
        logger.debug(f"解码完成，时长 {audio.duration}s，耗时 {elapsed:.5f}s (RTF: {elapsed / audio.duration})")
        return audio
    except av.error.InvalidDataError as e:
        raise HTTPException(
            status_code=415,
            detail="音频解码失败。不支持提供的文件类型。",
        ) from e
    except av.error.ValueError as e:
        raise HTTPException(
            status_code=400,
            detail="音频解码失败。提供的文件可能为空。",
        ) from e
    except Exception as e:
        logger.exception(
            "音频解码失败。这可能是一个错误，请提交 issue。"
        )
        raise HTTPException(status_code=500, detail="音频解码失败。") from e


# FastAPI 依赖项：音频文件
AudioFileDependency = Annotated[Audio, Depends(audio_file_dependency)]


@lru_cache
def get_completion_client() -> AsyncCompletions:
    """获取 OpenAI 兼容的聊天补全客户端。"""
    config = get_config()
    oai_client = AsyncOpenAI(
        base_url=config.chat_completion_base_url,
        api_key=config.chat_completion_api_key.get_secret_value(),
        max_retries=0,
    )
    return oai_client.chat.completions


async def get_completion_client_async() -> AsyncCompletions:
    return get_completion_client()


# FastAPI 依赖项：聊天补全客户端
CompletionClientDependency = Annotated[AsyncCompletions, Depends(get_completion_client_async)]


@lru_cache
def get_speech_client() -> AsyncSpeech:
    """获取 OpenAI 兼容的语音合成 (TTS) 客户端。"""
    config = get_config()
    if config.loopback_host_url is None:
        # 如果未设置回环地址，则通过 ASGI 传输直接调用内部路由
        from speaches.routers.speech import (
            router as speech_router,
        )

        http_client = AsyncClient(
            transport=ASGITransport(speech_router),
            base_url="http://test/v1",
        )
    else:
        http_client = AsyncClient(
            base_url=f"{config.loopback_host_url}/v1",
        )
    oai_client = AsyncOpenAI(
        http_client=http_client,
        api_key=config.api_key.get_secret_value() if config.api_key else "cant-be-empty",
        max_retries=0,
        base_url=f"{config.loopback_host_url}/v1" if config.loopback_host_url else None,
    )
    return oai_client.audio.speech


def get_speech_client_async() -> AsyncSpeech:
    return get_speech_client()


# FastAPI 依赖项：语音合成客户端
SpeechClientDependency = Annotated[AsyncSpeech, Depends(get_speech_client_async)]


@lru_cache
def get_transcription_client() -> AsyncTranscriptions:
    """获取 OpenAI 兼容的语音转文本 (STT) 客户端。"""
    config = get_config()
    if config.loopback_host_url is None:
        from speaches.routers.stt import (
            router as stt_router,
        )

        http_client = AsyncClient(
            transport=ASGITransport(stt_router),
            base_url="http://test/v1",
        )
    else:
        http_client = AsyncClient(
            base_url=f"{config.loopback_host_url}/v1",
        )
    oai_client = AsyncOpenAI(
        http_client=http_client,
        api_key=config.api_key.get_secret_value() if config.api_key else "cant-be-empty",
        max_retries=0,
        base_url=f"{config.loopback_host_url}/v1" if config.loopback_host_url else None,
    )
    return oai_client.audio.transcriptions


async def get_transcription_client_async() -> AsyncTranscriptions:
    return get_transcription_client()


# FastAPI 依赖项：语音转文本客户端
TranscriptionClientDependency = Annotated[AsyncTranscriptions, Depends(get_transcription_client_async)]
