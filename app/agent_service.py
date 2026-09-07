from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

from .schemas import AgentRequest, AgentResponse


SYSTEM_PROMPT = """你是山音伙伴的小音，只服务儿童音乐创作与教师现场引导。
不得索取、推断或复述姓名、学校、地址、电话等个人信息；不做诊断、排名或儿童画像。
只返回一个 JSON 对象，字段必须为 schemaVersion, scene, text；creation_story 可加 story，teacher_report 可加 report。
文案简短、具体、鼓励尝试，不宣称替代教师判断。"""

FALLBACKS = {
    "theme_intent": "选一个你想唱的心情，然后哼出一小段吧。",
    "melody_feedback": "我听见了你的旋律，我们把它慢慢长成一首歌。",
    "arrangement_variants": "试试三个版本，选最像你心里那一个。",
    "practice_feedback": "再听一遍这一句，然后用舒服的声音跟唱。",
    "creation_story": "这首歌从一小段哼唱出发，经过重复和变化，长成了完整作品。",
    "teacher_report": "本次结果只用于现场教学参考，请结合学生当下的真实表现判断。",
}


async def run_agent(request: AgentRequest) -> AgentResponse:
    if not os.getenv("DEEPSEEK_API_KEY", "").strip():
        return fallback(request, "offline")
    try:
        value = await asyncio.wait_for(asyncio.to_thread(_run_agentscope, request), timeout=18)
        return validate_model_result(request, value)
    except Exception:
        return fallback(request, "fallback")


def _run_agentscope(request: AgentRequest) -> dict[str, Any]:
    """沿用 0.4.2-local 的 AgentScope 1.x + DeepSeek 调用路径。

    模型只负责生成白名单文案；所有可执行音乐数据仍由本地算法和 Pydantic 契约决定。
    """
    from agentscope.agent import ReActAgent
    from agentscope.formatter import OpenAIChatFormatter
    from agentscope.memory import InMemoryMemory
    from agentscope.message import Msg
    from agentscope.model import OpenAIChatModel
    from agentscope.tool import Toolkit

    model = OpenAIChatModel(
        os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-v4-flash"),
        api_key=os.environ["DEEPSEEK_API_KEY"],
        stream=False,
        client_kwargs={
            "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/"),
            "timeout": 16,
            "max_retries": 0,
        },
        generate_kwargs={"response_format": {"type": "json_object"}},
    )

    async def invoke() -> dict[str, Any]:
        agent = ReActAgent(
            name="Xiaoyin",
            model=model,
            sys_prompt=SYSTEM_PROMPT,
            toolkit=Toolkit(),
            memory=InMemoryMemory(),
            formatter=OpenAIChatFormatter(),
        )
        payload = {
            "schemaVersion": "1",
            "scene": request.scene,
            "audience": request.audience,
            "context": request.context,
        }
        reply = await agent(Msg("user", json.dumps(payload, ensure_ascii=False), "user"))
        content = reply.get_text_content() if hasattr(reply, "get_text_content") else str(reply.content)
        return _extract_json(content)

    return asyncio.run(invoke())


def _extract_json(value: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", value.strip())
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("model result must be object")
    return parsed


def validate_model_result(request: AgentRequest, value: dict[str, Any]) -> AgentResponse:
    if value.get("schemaVersion") != "1" or value.get("scene") != request.scene:
        raise ValueError("agent protocol mismatch")
    allowed = {"schemaVersion", "scene", "text"}
    if request.scene == "creation_story":
        allowed.add("story")
    if request.scene == "teacher_report":
        allowed.add("report")
    cleaned = {key: value[key] for key in value if key in allowed}
    cleaned["source"] = "agent"
    return AgentResponse.model_validate(cleaned)


def fallback(request: AgentRequest, source: str) -> AgentResponse:
    kwargs: dict[str, Any] = {
        "scene": request.scene,
        "text": FALLBACKS[request.scene],
        "source": source,
    }
    if request.scene == "creation_story":
        kwargs["story"] = FALLBACKS[request.scene]
    if request.scene == "teacher_report":
        kwargs["report"] = FALLBACKS[request.scene]
    return AgentResponse.model_validate(kwargs)
