"""Allowlisted dispatch and a bounded trace; no language model or provider is called here."""

from typing import Any, Literal

from pydantic import Field, ValidationError

from forge.agents.tools import (
    Contract,
    EvidenceSearch,
    Inspection,
    InspectRecordingArgs,
    InvestigationTools,
    SearchEvidenceArgs,
)


class ToolCall(Contract):
    call_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=64)
    arguments: dict[str, Any]


class ToolError(Contract):
    code: Literal["unknown_tool", "invalid_arguments", "duplicate_call_id", "tool_failed"]
    message: str


class ToolResult(Contract):
    call_id: str
    tool: str
    result: Inspection | EvidenceSearch | None = None
    error: ToolError | None = None


class ToolBudgetExceeded(RuntimeError):
    """Stop the caller's loop; do not keep retrying tool calls."""


class ToolSession:
    def __init__(self, tools: InvestigationTools, *, max_calls=3):
        if isinstance(max_calls, bool) or not isinstance(max_calls, int) or not 1 <= max_calls <= 4:
            raise ValueError("Tool-call budget must be an integer from one to four.")
        self.tools = tools
        self.max_calls = max_calls
        self._trace = []
        self._ids = set()

    @property
    def trace(self):
        return tuple(item.model_copy(deep=True) for item in self._trace)

    def execute(self, call: ToolCall) -> ToolResult:
        if len(self._trace) >= self.max_calls:
            raise ToolBudgetExceeded("Tool-call budget exhausted.")
        handlers = {
            "inspect_recording": (InspectRecordingArgs, self.tools.inspect_recording),
            "search_evidence": (SearchEvidenceArgs, self.tools.search_evidence),
        }
        result, error = None, None
        if call.call_id in self._ids:
            error = ToolError(
                code="duplicate_call_id", message="Call IDs must be unique per session."
            )
        elif call.name not in handlers:
            error = ToolError(code="unknown_tool", message="This tool is not available.")
        else:
            schema, handler = handlers[call.name]
            try:
                args = schema.model_validate(call.arguments)
            except ValidationError:
                error = ToolError(
                    code="invalid_arguments", message="Arguments do not match the tool schema."
                )
            else:
                try:
                    result = handler(args)
                    if len(result.model_dump_json()) > 24000:
                        raise ValueError("Tool response exceeds the local character limit.")
                except (OSError, ValueError, KeyError, ImportError):
                    result = None
                    error = ToolError(
                        code="tool_failed",
                        message="The local tool could not return a verified result. Check the local data, model and corpus setup.",
                    )
        self._ids.add(call.call_id)
        response = ToolResult(call_id=call.call_id, tool=call.name, result=result, error=error)
        self._trace.append(response.model_copy(deep=True))
        return response
