"""CoreWeave Agent Lens OpenTelemetry instrumentation + offline MemoryTracer."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from .common import SUITE_VERSION


def make_conversation_id(batch_id: str, model_spec_id: str, effort: str, item_id: str) -> str:
    seed_str = f"agentnerf:{SUITE_VERSION}:{batch_id}:{model_spec_id}:{effort}:{item_id}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed_str))


class MemoryToolSpan:
    def __init__(self, name: str, arguments: str, tool_call_id: str) -> None:
        self.name = name
        self.arguments = arguments
        self.tool_call_id = tool_call_id
        self.attributes: dict[str, Any] = {}
        self.error: str | None = None

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        self.attributes.update(attributes)

    def record_error(self, error: BaseException) -> None:
        self.error = f"{type(error).__name__}: {error}"


class MemoryTurnHandle:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        self.payload["attributes"].update(attributes)

    def record_output(self, text: str) -> None:
        self.payload["output"] = text

    def record_error(self, error: BaseException) -> None:
        self.payload["error"] = f"{type(error).__name__}: {error}"

    @contextmanager
    def tool(self, *, name: str, arguments: dict[str, Any], tool_call_id: str) -> Iterator[MemoryToolSpan]:
        span = MemoryToolSpan(name, json.dumps(arguments, sort_keys=True), tool_call_id)
        try:
            yield span
        finally:
            self.payload["tools"].append(
                {
                    "name": span.name,
                    "tool_call_id": span.tool_call_id,
                    "arguments": span.arguments,
                    "attributes": dict(span.attributes),
                    "error": span.error,
                }
            )


class MemoryTracer:
    """In-memory tracer for unit tests and --no-agentlens offline runs."""

    def __init__(self) -> None:
        self.conversations: list[dict[str, Any]] = []

    def flush(self) -> None:
        return None

    @contextmanager
    def conversation(
        self,
        *,
        agent_name: str,
        model: str,
        conversation_id: str,
        conversation_name: str,
        attributes: dict[str, Any],
    ) -> Iterator[dict[str, Any]]:
        conv = {
            "agent_name": agent_name,
            "model": model,
            "conversation_id": conversation_id,
            "conversation_name": conversation_name,
            "attributes": dict(attributes),
            "turns": [],
        }
        self.conversations.append(conv)
        yield conv

    @contextmanager
    def turn(
        self,
        conv: dict[str, Any],
        *,
        user_message: str,
        agent_name: str,
        agent_version: str,
        model: str,
        system_instructions: list[str],
        tag: str,
        attributes: dict[str, Any],
    ) -> Iterator[MemoryTurnHandle]:
        payload = {
            "user_message": user_message,
            "agent_name": agent_name,
            "agent_version": agent_version,
            "model": model,
            "system_instructions": list(system_instructions),
            "attributes": {**conv["attributes"], **attributes, "weave.tag": tag},
            "tools": [],
            "output": "",
            "error": None,
        }
        handle = MemoryTurnHandle(payload)
        try:
            yield handle
        finally:
            conv["turns"].append(payload)


class AgentLensTurnHandle:
    def __init__(self, sdk_turn: Any, tag: str) -> None:
        self.sdk_turn = sdk_turn
        self.tag = tag

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        clean = {k: v for k, v in attributes.items() if v is not None}
        self.sdk_turn.set_attributes(clean)

    def record_output(self, text: str) -> None:
        if not text:
            return
        from coreweave.forge.agentlens.tracing import Message

        self.sdk_turn.record(output_messages=[Message(role="assistant", content=text)])

    def record_error(self, error: BaseException) -> None:
        self.sdk_turn.record_error(error)

    @contextmanager
    def tool(self, *, name: str, arguments: dict[str, Any], tool_call_id: str) -> Iterator[Any]:
        arg_json = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
        with self.sdk_turn.start_tool(
            name=name,
            arguments=arg_json,
            tool_call_id=tool_call_id,
        ) as span:
            span.set_attributes({"weave.tag": self.tag})
            yield span


class AgentLensTracer:
    """Live OpenTelemetry exporter targeting CoreWeave Agent Lens (`https://trace.wandb.ai`)."""

    def __init__(self, project_name: str, *, api_key: str | None = None, base_url: str | None = None) -> None:
        from coreweave.forge.agentlens import tracing

        tracing.init(
            project_name,
            api_key=api_key,
            base_url=base_url,
            service_name="agentnerf",
            autopatch_integrations=True,
        )
        self.tracing = tracing
        self.project_name = project_name

    def flush(self) -> None:
        self.tracing.force_flush()

    @contextmanager
    def conversation(
        self,
        *,
        agent_name: str,
        model: str,
        conversation_id: str,
        conversation_name: str,
        attributes: dict[str, Any],
    ) -> Iterator[Any]:
        clean = {k: v for k, v in attributes.items() if v is not None}
        conv = self.tracing.start_conversation(
            agent_name=agent_name,
            model=model,
            conversation_id=conversation_id,
            conversation_name=conversation_name,
            attributes=clean,
        )
        try:
            yield conv
        finally:
            conv.end()

    @contextmanager
    def turn(
        self,
        conv: Any,
        *,
        user_message: str,
        agent_name: str,
        agent_version: str,
        model: str,
        system_instructions: list[str],
        tag: str,
        attributes: dict[str, Any],
    ) -> Iterator[AgentLensTurnHandle]:
        sdk_turn = conv.start_turn(
            user_message=user_message,
            model=model,
            agent_name=agent_name,
            agent_version=agent_version,
            system_instructions=system_instructions,
        )
        clean = {k: v for k, v in attributes.items() if v is not None}
        clean["weave.tag"] = tag
        sdk_turn.set_attributes(clean)
        try:
            yield AgentLensTurnHandle(sdk_turn, tag)
        finally:
            sdk_turn.end()
