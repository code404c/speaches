from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
import gc
import logging
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from speaches.config import OrtOptions

logger = logging.getLogger(__name__)


def get_ort_providers_with_options(ort_opts: OrtOptions) -> list[tuple[str, dict]]:
    """
    根据配置获取可用的 ONNX Runtime 提供程序及其选项。
    按照优先级排序，并排除被明确禁用的提供程序。
    """
    from onnxruntime import get_available_providers  # pyright: ignore[reportAttributeAccessIssue]

    available_providers: list[str] = get_available_providers()
    logger.debug(f"可用的 ONNX Runtime 提供程序: {available_providers}")
    # 排除配置中指定的提供程序
    available_providers = [provider for provider in available_providers if provider not in ort_opts.exclude_providers]
    # 根据优先级排序，值越大优先级越高
    available_providers = sorted(
        available_providers,
        key=lambda x: ort_opts.provider_priority.get(x, 0),
        reverse=True,
    )
    # 组合提供程序名称及其特定的配置选项
    available_providers_with_opts = [
        (provider, ort_opts.provider_opts.get(provider, {})) for provider in available_providers
    ]
    logger.debug(f"最终使用的 ONNX Runtime 提供程序: {available_providers_with_opts}")
    return available_providers_with_opts


class SelfDisposingModel[T]:
    """
    自动释放模型包装器。
    利用引用计数和 TTL (生存时间) 机制管理模型的加载与卸载，以节省显存/内存。
    支持上下文管理器协议 (__enter__/__exit__)。
    """
    def __init__(
        self,
        model_id: str,
        load_fn: Callable[[], T],
        ttl: int,
        model_unloaded_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.model_id = model_id
        self.load_fn = load_fn  # 模型加载函数
        self.ttl = ttl          # 模型闲置后的存活时间（秒）
        self.model_unloaded_callback = model_unloaded_callback

        self.ref_count: int = 0  # 引用计数
        self.rlock = threading.RLock()
        self.expire_timer: threading.Timer | None = None
        self.model: T | None = None

    def unload(self) -> None:
        """卸载模型，释放资源。"""
        with self.rlock:
            if self.model is None:
                raise ValueError(f"模型 {self.model_id} 未加载。{self.ref_count=}")
            if self.ref_count > 0:
                raise ValueError(f"模型 {self.model_id} 仍在被使用。{self.ref_count=}")
            if self.expire_timer:
                self.expire_timer.cancel()
            self.model = None
            gc.collect()  # 触发垃圾回收
            logger.info(f"模型 {self.model_id} 已卸载")
            if self.model_unloaded_callback is not None:
                self.model_unloaded_callback(self.model_id)

    def _load(self) -> None:
        """执行模型加载逻辑。"""
        with self.rlock:
            assert self.model is None
            logger.debug(f"正在加载模型 {self.model_id}")
            start = time.perf_counter()
            self.model = self.load_fn()
            logger.info(f"模型 {self.model_id} 加载完成，耗时 {time.perf_counter() - start:.2f}s")

    def _increment_ref(self) -> None:
        """增加引用计数，并取消可能的卸载定时器。"""
        with self.rlock:
            self.ref_count += 1
            if self.expire_timer:
                logger.debug(f"模型原本设置为 {self.expire_timer.interval}s 后过期，现已取消")
                self.expire_timer.cancel()
            logger.debug(f"增加 {self.model_id} 的引用计数，当前: {self.ref_count=}")

    def _decrement_ref(self) -> None:
        """减少引用计数。如果计数归零，则根据 TTL 设置卸载任务。"""
        with self.rlock:
            self.ref_count -= 1
            logger.debug(f"减少 {self.model_id} 的引用计数，当前: {self.ref_count=}")
            if self.ref_count <= 0:
                if self.ttl > 0:
                    logger.debug(f"模型 {self.model_id} 处于闲置状态，计划在 {self.ttl}s 后卸载")
                    self.expire_timer = threading.Timer(self.ttl, self.unload)
                    self.expire_timer.start()
                elif self.ttl == 0:
                    logger.info(f"模型 {self.model_id} 处于闲置状态，立即卸载")
                    self.unload()
                else:
                    # TTL 为 -1 时不卸载
                    logger.info(f"模型 {self.model_id} 处于闲置状态，根据设置不执行卸载")

    def __enter__(self) -> T:
        """进入上下文：确保模型已加载并增加引用计数。"""
        with self.rlock:
            if self.model is None:
                self._load()
            self._increment_ref()
            assert self.model is not None
            return self.model

    def __exit__(self, *_args) -> None:
        """退出上下文：减少引用计数。"""
        self._decrement_ref()


class BaseModelManager[T](ABC):
    """
    模型管理器抽象基类。
    负责管理一组 `SelfDisposingModel` 实例。
    """
    def __init__(self, ttl: int) -> None:
        self.ttl = ttl
        self.loaded_models: OrderedDict[str, SelfDisposingModel[T]] = OrderedDict()
        self._lock = threading.Lock()

    @abstractmethod
    def _load_fn(self, model_id: str) -> T:
        """子类需实现具体的模型加载逻辑。"""
        pass

    def _handle_model_unloaded(self, model_id: str) -> None:
        """当模型被卸载时，从已加载列表中移除。"""
        with self._lock:
            if model_id in self.loaded_models:
                del self.loaded_models[model_id]

    def unload_model(self, model_id: str) -> None:
        """手动卸载指定模型。"""
        with self._lock:
            model = self.loaded_models.get(model_id)
            if model is None:
                raise KeyError(f"未找到模型 {model_id}")
            del self.loaded_models[model_id]
        model.unload()

    def load_model(self, model_id: str) -> SelfDisposingModel[T]:
        """加载或获取已存在的模型包装器。"""
        with self._lock:
            if model_id in self.loaded_models:
                logger.debug(f"模型 {model_id} 已经存在于管理器中")
                return self.loaded_models[model_id]
            self.loaded_models[model_id] = SelfDisposingModel[T](
                model_id,
                load_fn=lambda: self._load_fn(model_id),
                ttl=self.ttl,
                model_unloaded_callback=self._handle_model_unloaded,
            )
            return self.loaded_models[model_id]
