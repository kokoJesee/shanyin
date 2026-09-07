from __future__ import annotations

import json
import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from pydantic import ValidationError
from pydantic_core import ValidationError as CoreValidationError

from .agent_service import run_agent
from .render_service import render_arrangement, resolve_local_file
from .schemas import AgentRequest, RenderRequest, RenderResponse


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("shanyin")
app = FastAPI(title="Shanyin CloudRun API", version="1.0.0", docs_url=None, redoc_url=None)


@app.get("/health")
async def health() -> dict[str, str]:
    # 固定响应；绝不回显请求头、CloudBase 上下文或环境变量。
    return {"status": "ok", "service": "shanyin-agent", "schemaVersion": "1"}


@app.post("/api/v1/agent")
async def agent_endpoint(payload: AgentRequest) -> dict[str, Any]:
    result = await run_agent(payload)
    logger.info("agent request_id=%s scene=%s source=%s", payload.requestId, payload.scene, result.source)
    return result.model_dump(exclude_none=True)


@app.post("/api/v1/render", response_model=RenderResponse)
async def render_endpoint(payload: RenderRequest, request: Request) -> RenderResponse:
    try:
        # 用 PUBLIC_BASE_URL 环境变量强制 https，避免云托管 Host header 给出 http://
        public_base_url = os.getenv("PUBLIC_BASE_URL") or f"https://{request.headers.get('host', request.url.netloc)}"
        result = render_arrangement(payload.arrangement, payload.accompanimentGain, public_base_url)
        logger.info("render request_id=%s status=ok duration=%.2f", payload.requestId, payload.arrangement.duration)
        return result
    except ValidationError as exc:
        # 422 时打印完整 payload + 错误详情，CloudBase 日志可直接看到错哪个字段
        logger.warning("render 422 request_id=%s errors=%s payload=%s", payload.requestId, exc.errors(), payload.model_dump())
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    except Exception as exc:
        logger.exception("render request_id=%s status=failed", payload.requestId)
        raise HTTPException(status_code=503, detail="render unavailable") from exc


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """FastAPI 在 Pydantic 校验失败时也会走这里，把 detail 写到日志"""
    logger.warning("global 422 errors=%s body=%s", exc.errors(), (await request.body()).decode("utf-8", errors="replace")[:2000])
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.get("/api/v1/render/files/{file_id}")
async def download_render(file_id: str) -> FileResponse:
    path = resolve_local_file(file_id)
    if not path:
        raise HTTPException(status_code=404, detail="file expired or missing")
    return FileResponse(path, media_type="audio/mpeg", filename="shanyin-work.mp3")


@app.post("/process")
async def legacy_process(payload: dict[str, Any]) -> dict[str, Any]:
    """兼容 0.4.2-local Web 版的 /process 输入，统一走新版白名单响应。"""
    try:
        text = payload["input"][0]["content"][0]["text"]
        legacy = json.loads(text)
        request = AgentRequest(
            schemaVersion="1",
            requestId=_safe_id(payload.get("session_id"), "request"),
            sessionId=_safe_id(payload.get("session_id"), "session"),
            scene=legacy["scene"],
            audience=legacy["audience"],
            context=legacy["context"],
        )
    except (KeyError, IndexError, TypeError, json.JSONDecodeError, ValidationError) as exc:
        raise HTTPException(status_code=422, detail="invalid legacy request") from exc
    return (await run_agent(request)).model_dump(exclude_none=True)


def _safe_id(value: Any, prefix: str) -> str:
    raw = "".join(character for character in str(value or "") if character.isalnum() or character in "_-")[:96]
    return raw if len(raw) >= 8 else f"{prefix}-anonymous"
