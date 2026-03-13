from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os
from typing import TYPE_CHECKING
import uuid

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Response,
)
from fastapi.exception_handlers import (
    http_exception_handler,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import RedirectResponse

from speaches.dependencies import ApiKeyDependency, get_config, get_executor_registry
from speaches.logger import setup_logger
from speaches.routers.chat import (
    router as chat_router,
)
from speaches.routers.diarization import (
    router as diarization_router,
)
from speaches.routers.misc import (
    public_router as misc_public_router,
)
from speaches.routers.misc import (
    router as misc_router,
)
from speaches.routers.models import (
    router as models_router,
)
from speaches.routers.realtime_rtc import (
    router as realtime_rtc_router,
)
from speaches.routers.realtime_ws import (
    router as realtime_ws_router,
)
from speaches.routers.speech import (
    router as speech_router,
)
from speaches.routers.speech_embedding import (
    router as speech_embedding_router,
)
from speaches.routers.stt import (
    router as stt_router,
)
from speaches.routers.vad import (
    router as vad_router,
)
from speaches.utils import APIProxyError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

# OpenAPI 标签元数据，用于 Swagger UI 的分组和说明
# 参考：https://swagger.io/docs/specification/v3_0/grouping-operations-with-tags/
TAGS_METADATA = [
    {"name": "automatic-speech-recognition", "description": "自动语音识别 (ASR)"},
    {"name": "speech-to-text", "description": "语音转文本 (STT)"},
    {"name": "speaker-embedding", "description": "说话人嵌入 (Speaker Embedding)"},
    {"name": "realtime", "description": "实时处理 (Realtime)"},
    {"name": "models", "description": "模型管理"},
    {"name": "diagnostic", "description": "诊断与健康检查"},
    {
        "name": "experimental",
        "description": "实验性功能，暂不建议公开使用。可能随时更改或移除。",
    },
]


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """管理应用程序的生命周期。
    在启动时执行模型预加载，在退出时执行清理逻辑。
    """
    logger = logging.getLogger(__name__)
    config = get_config()

    # 如果配置了预加载模型，则在启动时下载
    if config.preload_models:
        logger.info(f"启动时正在预加载 {len(config.preload_models)} 个模型")
        executor_registry = get_executor_registry()

        for model_id in config.preload_models:
            logger.info(f"正在下载模型: {model_id}")
            executor_registry.download_model_by_id(model_id)
            logger.info(f"成功下载模型: {model_id}")

    yield


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用程序实例。"""
    config = get_config()
    setup_logger(config.log_level)  # 设置日志系统
    logger = logging.getLogger(__name__)

    logger.debug(f"当前配置: {config}")

    # 如果配置了 OTLP 端点，则初始化 OpenTelemetry 遥测
    if config.otel_exporter_otlp_endpoint:
        from speaches.tracing import setup_telemetry

        setup_telemetry(config.otel_exporter_otlp_endpoint, config.otel_service_name)

        # 自动对常用库进行埋点
        from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        from opentelemetry.instrumentation.logging import LoggingInstrumentor

        AsyncioInstrumentor().instrument()
        HTTPXClientInstrumentor().instrument()
        LoggingInstrumentor().instrument()

    # 创建主应用实例，暂不设置全局认证
    app = FastAPI(
        title="Speaches",
        version="0.8.3",  # TODO: 发布时更新版本号
        license_info={"name": "MIT License", "identifier": "MIT"},
        openapi_tags=TAGS_METADATA,
        lifespan=lifespan,
    )

    # 如果启用了遥测，则对 FastAPI 应用进行埋点
    if config.otel_exporter_otlp_endpoint:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)

    # 注册 APIProxyError 的全局异常处理器
    @app.exception_handler(APIProxyError)
    async def _api_proxy_error_handler(_request: Request, exc: APIProxyError) -> JSONResponse:
        error_id = str(uuid.uuid4())
        logger.exception(f"[{error_id}] {exc.message}")
        content = {
            "detail": exc.message,
            "hint": exc.hint,
            "suggested_fixes": exc.suggestions,
            "error_id": error_id,
        }

        # 如果是 DEBUG 模式且存在调试信息，则包含在返回内容中
        log_level = os.getenv("SPEACHES_LOG_LEVEL", "INFO").upper()
        if log_level == "DEBUG" and exc.debug:
            content["debug"] = exc.debug
        return JSONResponse(status_code=exc.status_code, content=content)

    @app.exception_handler(StarletteHTTPException)
    async def _custom_http_exception_handler(request: Request, exc: HTTPException) -> Response:
        """自定义 HTTP 异常处理器，记录错误日志。"""
        logger.error(f"HTTP 错误: {exc}")
        return await http_exception_handler(request, exc)

    # 包含不需要身份验证的公开路由
    app.include_router(misc_public_router)

    # 如果配置了 API 密钥，则为以下路由添加身份验证依赖
    http_dependencies = []
    if config.api_key is not None:
        http_dependencies.append(ApiKeyDependency)

    # 挂载业务路由
    app.include_router(chat_router, dependencies=http_dependencies)
    app.include_router(stt_router, dependencies=http_dependencies)
    app.include_router(models_router, dependencies=http_dependencies)
    app.include_router(misc_router, dependencies=http_dependencies)
    app.include_router(realtime_rtc_router, dependencies=http_dependencies)
    app.include_router(speech_router, dependencies=http_dependencies)
    app.include_router(speech_embedding_router, dependencies=http_dependencies)
    app.include_router(vad_router, dependencies=http_dependencies)
    app.include_router(diarization_router, dependencies=http_dependencies)

    # WebSocket 路由（由其自身处理身份验证）
    app.include_router(realtime_ws_router)

    # 挂载实时控制台静态文件
    app.get("/v1/realtime", include_in_schema=False)(lambda: RedirectResponse(url="/v1/realtime/"))
    app.mount("/v1/realtime", StaticFiles(directory="realtime-console/dist", html=True))

    # 配置 CORS（跨域资源共享）中间件
    if config.allow_origins is not None:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.allow_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # 如果启用了 UI，则挂载 Gradio 界面
    if config.enable_ui:
        import gradio as gr

        from speaches.ui.app import create_gradio_demo

        app = gr.mount_gradio_app(app, create_gradio_demo(config), path="")

        logger = logging.getLogger("speaches.main")
        if config.host and config.port:
            display_host = "localhost" if config.host in ("0.0.0.0", "127.0.0.1") else config.host
            url = f"http://{display_host}:{config.port}/"
            logger.info(f"\n\n访问 Speaches 的 Gradio Web UI，请在浏览器中打开：\n\n{url}\n\n")

    return app
