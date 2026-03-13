from __future__ import annotations

from typing import TYPE_CHECKING

from speaches.executors.pyannote_speaker_segmentation import (
    PyannoteSpeakerSegmentationModelManager,
    pyannote_speaker_segmentation_model_registry,
)

if TYPE_CHECKING:
    from speaches.config import Config

from speaches.executors.kokoro import KokoroModelManager, kokoro_model_registry
from speaches.executors.parakeet import ParakeetModelManager, parakeet_model_registry
from speaches.executors.piper import PiperModelManager, piper_model_registry
from speaches.executors.shared.executor import Executor
from speaches.executors.silero_vad_v5 import SileroVADModelManager, silero_vad_model_registry
from speaches.executors.wespeaker_speaker_embedding import (
    WespeakerSpeakerEmbeddingModelManager,
    wespeaker_speaker_embedding_model_registry,
)
from speaches.executors.whisper import WhisperModelManager, whisper_model_registry


class ExecutorRegistry:
    """
    模型执行器注册表。
    负责初始化并持有所有的模型执行器，并根据任务类型（转录、翻译、TTS等）提供相应的执行器。
    """
    def __init__(self, config: Config) -> None:
        # 初始化 Whisper 执行器（用于 ASR 和 翻译）
        self._whisper_executor = Executor(
            name="whisper",
            model_manager=WhisperModelManager(config.stt_model_ttl, config.whisper),
            model_registry=whisper_model_registry,
            task="automatic-speech-recognition",
        )
        # 初始化 Parakeet 执行器（ASR 备选）
        self._parakeet_executor = Executor(
            name="parakeet",
            model_manager=ParakeetModelManager(config.stt_model_ttl, config.unstable_ort_opts),
            model_registry=parakeet_model_registry,
            task="automatic-speech-recognition",
        )
        # 初始化 Piper 执行器（TTS）
        self._piper_executor = Executor(
            name="piper",
            model_manager=PiperModelManager(config.tts_model_ttl, config.unstable_ort_opts),
            model_registry=piper_model_registry,
            task="text-to-speech",
        )
        # 初始化 Kokoro 执行器（高性能 TTS）
        self._kokoro_executor = Executor(
            name="kokoro",
            model_manager=KokoroModelManager(config.tts_model_ttl, config.unstable_ort_opts),
            model_registry=kokoro_model_registry,
            task="text-to-speech",
        )
        # 初始化 WeSpeaker 说话人嵌入执行器
        self._wespeaker_speaker_embedding_executor = Executor(
            name="wespeaker-speaker-embedding",
            model_manager=WespeakerSpeakerEmbeddingModelManager(0, config.unstable_ort_opts),  # 注意：TTL 硬编码为 0
            model_registry=wespeaker_speaker_embedding_model_registry,
            task="speaker-embedding",
        )
        # 初始化 Pyannote 说话人分割执行器
        self._pyannote_speaker_segmentation_executor = Executor(
            name="pyannote-speaker-segmentation",
            model_manager=PyannoteSpeakerSegmentationModelManager(0, config.unstable_ort_opts),  # 注意：TTL 硬编码为 0
            model_registry=pyannote_speaker_segmentation_model_registry,
            task="voice-activity-detection",
        )
        # 初始化 VAD 执行器（语音活动检测）
        self._vad_executor = Executor(
            name="vad",
            model_manager=SileroVADModelManager(config.vad_model_ttl, config.unstable_ort_opts),
            model_registry=silero_vad_model_registry,
            task="voice-activity-detection",
        )

    @property
    def transcription(self):  # noqa: ANN201
        """返回支持转录任务的所有执行器。"""
        return (self._whisper_executor, self._parakeet_executor)

    @property
    def translation(self):  # noqa: ANN201
        """返回支持翻译任务的所有执行器。"""
        return (self._whisper_executor,)

    @property
    def text_to_speech(self):  # noqa: ANN201
        """返回支持语音合成 (TTS) 任务的所有执行器。"""
        return (self._piper_executor, self._kokoro_executor)

    @property
    def speaker_embedding(self):  # noqa: ANN201
        """返回说话人嵌入执行器。"""
        return (self._wespeaker_speaker_embedding_executor,)

    @property
    def speaker_segmentation(self):  # noqa: ANN201
        """返回说话人分割执行器。"""
        return (self._pyannote_speaker_segmentation_executor,)

    @property
    def vad(self):  # noqa: ANN201
        """返回 VAD (语音活动检测) 执行器。"""
        return self._vad_executor

    def all_executors(self):  # noqa: ANN201
        """返回所有的执行器列表。"""
        return (
            self._whisper_executor,
            self._parakeet_executor,
            self._piper_executor,
            self._kokoro_executor,
            self._wespeaker_speaker_embedding_executor,
            self._pyannote_speaker_segmentation_executor,
            self._vad_executor,
        )

    def download_model_by_id(self, model_id: str) -> bool:
        """根据模型 ID 自动在所有执行器的注册表中查找并下载模型文件。"""
        for executor in self.all_executors():
            if model_id in [model.id for model in executor.model_registry.list_remote_models()]:
                return executor.model_registry.download_model_files_if_not_exist(model_id)
        raise ValueError(f"未找到模型 '{model_id}'")
