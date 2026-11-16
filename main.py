"""free-model-aggregator-02: one OpenAI/Anthropic front for OpenRouter's free models.

The entry is a single-file Python 3.13 server. It reads the trial brief, registers with
the env, serves both completion wires plus GET /v1/models, and answers every prompt with
a real completion dialled through the grader's gateway. Model choice is deterministic:
rank the brief's free models by the house's rating, filtered to what the prompt needs,
with per-model cooldowns on refusals and failover across the rest. Nothing reads the
clock, so a live pass and its replay route identically.
"""

from __future__ import annotations

import codecs
import itertools
import json
import math
import os
import signal
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

Json = dict[str, object]

# ---------------------------------------------------------------- trial wiring

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
BRIEF_PATH = os.environ.get("MAKEFUN_PROMPT_FILE", "")
GATEWAY_ROOT = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
GATEWAY_KEY = os.environ.get("OPENAI_API_KEY", "")

PROTOCOL_VERSION = 1
WORK_POLL_SECONDS = 0.1
MAX_BODY_BYTES = 8 * 1024 * 1024
GATEWAY_TIMEOUT_SECONDS = 30
REGISTER_TIMEOUT_SECONDS = 5
MAX_TRACKED_FAILURES = 8
MAX_CALLS_PER_TRIAL = 60
PORT_FORBIDDEN = 8080

MODELS_DOC_SCHEMA = "free-model-aggregator-02/models/v1"
PROMPTS_DOC_SCHEMA = "free-model-aggregator-02/prompts/v1"
SCHEMA_FAMILY = "free-model-aggregator-02/"

PROVIDER_ID = "openrouter"
PROVIDER_PUBLIC_BASE = "https://openrouter.ai/api/v1"
PROVIDER_KEY_ENV = "OPENROUTER_API_KEY"

OPENAI_LIST_CREATED = 1767225600
ANTHROPIC_LIST_CREATED_AT = "2026-01-01T00:00:00Z"


def log(line: str) -> None:
    print(f"aggregator: {line}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------- JSON guards

def is_obj(value: object) -> bool:
    return isinstance(value, dict)


def is_int_float(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def as_str(value: object) -> str:
    return value if isinstance(value, str) else ""


def as_num(value: object) -> int | float | None:
    return value if is_int_float(value) else None


def as_num_or(value: object, fallback: int | float) -> int | float:
    return value if is_int_float(value) else fallback


def as_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


# ---------------------------------------------------------------- brief scanning

def json_objects_in(text: str) -> list[Json]:
    """Every {...} in the brief that parses and names a `schema` string, in order."""
    found: list[Json] = []
    cursor = text.find("{")
    while cursor != -1:
        closing = _balanced_end(text, cursor)
        if closing != -1:
            try:
                parsed = json.loads(text[cursor : closing + 1])
            except (json.JSONDecodeError, ValueError):
                parsed = None
            if is_obj(parsed) and isinstance(parsed.get("schema"), str):
                found.append(parsed)
        cursor = text.find("{", cursor + 1)
    return found


def _balanced_end(text: str, opening: int) -> int:
    """Index of the `}` matching the `{` at `opening`; braces inside strings are skipped."""
    depth = 0
    inside_string = False
    index = opening
    while index < len(text):
        char = text[index]
        if inside_string:
            if char == "\\":
                index += 2
                continue
            if char == '"':
                inside_string = False
        elif char == '"':
            inside_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return -1


def doc_with_array(docs: list[Json], keys: tuple[str, ...]) -> tuple[Json, str] | None:
    """The first contest document holding one of `keys` as an array."""
    for doc in docs:
        schema = as_str(doc.get("schema"))
        if SCHEMA_FAMILY not in schema:
            continue
        for key in keys:
            if isinstance(doc.get(key), list):
                return doc, key
    return None


def model_row(raw: object) -> Json | None:
    """One brief model row normalized for routing and the catalog."""
    if not is_obj(raw):
        return None
    slug = as_str(raw.get("id")) or as_str(raw.get("model"))
    if slug == "":
        return None
    vendor = as_str(raw.get("provider")) or as_str(raw.get("wire")) or "openrouter"
    gateway_id = slug if slug.lower().startswith("openrouter/") else (
        f"openrouter/{slug}" if vendor == PROVIDER_ID else slug
    )
    return {
        "id": gateway_id,
        "vendor": vendor,
        "name": as_str(raw.get("display_name")) or slug,
        "rating_milli": as_num_or(raw.get("rating_milli"), as_num_or(raw.get("rating"), 0)),
        "context_length": as_num(raw.get("context_length")) or as_num(raw.get("context")),
        "tools": bool(raw.get("tools")) if raw.get("tools") is not None else True,
        "vision": bool(raw.get("vision")) if raw.get("vision") is not None else False,
        "json": bool(raw.get("json")) if raw.get("json") is not None else True,
        "free": raw.get("free") is not False,
    }


def prompt_row(raw: object) -> Json | None:
    """One brief prompt row: wire, streaming, weight and the capabilities it demands."""
    if not is_obj(raw):
        return None
    prompt_id = as_str(raw.get("id"))
    if prompt_id == "":
        return None
    demands = raw.get("requires") if is_obj(raw.get("requires")) else {}
    wanted: Json = {}
    for capability in ("tools", "vision", "json"):
        if demands.get(capability) is True:
            wanted[capability] = True
    floor = as_num(demands.get("min_context"))
    if floor is not None:
        wanted["min_context"] = floor
    return {
        "id": prompt_id,
        "wire": "anthropic" if raw.get("api") == "anthropic" else "openai",
        "stream": raw.get("stream") is True,
        "weight": 2 if raw.get("weight") == 2 else 1,
        "requires": wanted,
    }


def load_brief(path: str) -> Json:
    """The brief's two documents, or an error note when either is missing."""
    if path == "":
        return {"models": [], "prompts": [], "fault": "MAKEFUN_PROMPT_FILE is not set"}
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except OSError as err:
        return {"models": [], "prompts": [], "fault": f"cannot read {path}: {err}"}
    docs = json_objects_in(text)
    models_doc = doc_with_array(docs, ("models", "ratings"))
    if models_doc is None:
        return {"models": [], "prompts": [], "fault": f"the brief holds no {MODELS_DOC_SCHEMA} document"}
    models = [row for row in (model_row(v) for v in models_doc[0][models_doc[1]]) if row is not None]
    prompts_doc = doc_with_array(docs, ("prompts", "manifest"))
    if prompts_doc is None:
        return {"models": models, "prompts": [], "fault": f"the brief holds no {PROMPTS_DOC_SCHEMA} document"}
    prompts = [row for row in (prompt_row(v) for v in prompts_doc[0][prompts_doc[1]]) if row is not None]
    return {"models": models, "prompts": prompts, "fault": None}


def free_pool(brief: Json) -> list[Json]:
    """The brief's free models, deduplicated, in the brief's order."""
    pool: list[Json] = []
    seen: set[str] = set()
    for row in brief["models"]:
        if row["free"] and row["id"] not in seen:
            seen.add(row["id"])
            pool.append(row)
    return pool


def model_capabilities(model: Json) -> Json:
    return {
        "context_length": model["context_length"],
        "tools": model["tools"],
        "vision": model["vision"],
        "json": model["json"],
    }


def satisfies(demands: Json, caps: Json) -> bool:
    """An unknown or false capability never satisfies a demand for it."""
    for capability in ("tools", "vision", "json"):
        if demands.get(capability) is True and caps[capability] is not True:
            return False
    floor = demands.get("min_context")
    if floor is not None and (caps["context_length"] is None or caps["context_length"] < floor):
        return False
    return True


def catalog_document(brief: Json) -> Json:
    """The register catalog: one provider, exactly the free ids the brief names."""
    rows: list[Json] = []
    seen: set[str] = set()
    for model in free_pool(brief):
        seen.add(model["id"])
        rows.append({"id": model["id"], "free": True, "context_length": model["context_length"]})
    if not rows:
        return {"providers": []}
    return {
        "providers": [
            {
                "id": PROVIDER_ID,
                "base_url": PROVIDER_PUBLIC_BASE,
                "key_env": PROVIDER_KEY_ENV,
                "wire": "openai",
                "models": rows,
            }
        ]
    }


BRIEF = load_brief(BRIEF_PATH)
CATALOG = catalog_document(BRIEF)
POOL = free_pool(BRIEF)

# The brief's prompts arrive in its order; a request outside the declared workload is
# read off its own body instead.
arrival_index = 0
arrival_lock = threading.Lock()


# ---------------------------------------------------------------- wire shapes

def openai_error_body(message: str, kind: str, code: str | None) -> Json:
    return {"error": {"message": message, "type": kind, "code": code}}


def anthropic_error_body(kind: str, message: str) -> Json:
    return {"type": "error", "error": {"type": kind, "message": message}}


def openai_listing(pool: list[Json]) -> Json:
    """GET /v1/models under an Authorization header: OpenAI's list shape."""
    return {
        "object": "list",
        "data": [
            {
                "id": model["id"],
                "object": "model",
                "created": OPENAI_LIST_CREATED,
                "owned_by": model.get("vendor", PROVIDER_ID),
            }
            for model in free_pool(BRIEF)
        ],
    }


def anthropic_listing(pool: list[Json]) -> Json:
    """GET /v1/models under an x-api-key header: Anthropic's list shape."""
    return {
        "data": [
            {
                "type": "model",
                "id": model["id"],
                "display_name": model.get("name", model["id"]),
                "created_at": ANTHROPIC_LIST_CREATED_AT,
            }
            for model in free_pool(BRIEF)
        ],
        "has_more": False,
    }


def text_in(content: object) -> str:
    """The text of a string body or of an array's text blocks; '' when there is none."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    pieces = [
        block["text"]
        for block in content
        if is_obj(block) and block.get("type") == "text" and isinstance(block.get("text"), str)
    ]
    return "\n".join(pieces)


def image_part(block: Json) -> Json | None:
    """One Anthropic image block as an OpenAI image_url part; None for anything else."""
    source = block.get("source")
    if not is_obj(source):
        return None
    if source.get("type") == "base64" and isinstance(source.get("data"), str):
        media = source["media_type"] if isinstance(source.get("media_type"), str) else "image/png"
        return {"type": "image_url", "image_url": {"url": f"data:{media};base64,{source['data']}"}}
    if source.get("type") == "url" and isinstance(source.get("url"), str):
        return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def content_part(block: Json) -> Json | None:
    if block.get("type") == "text" and isinstance(block.get("text"), str):
        return {"type": "text", "text": block["text"]}
    if block.get("type") == "image":
        return image_part(block)
    return None


def tool_result_text(content: object) -> str:
    """A tool_result block's content as text: its text, or its JSON when it holds something else."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = text_in(content)
        return text if text != "" else json.dumps(content, separators=(",", ":"), ensure_ascii=False)
    if content is None:
        return ""
    return json.dumps(content, separators=(",", ":"), ensure_ascii=False)


def tool_arguments(value: object) -> Json:
    """The gateway's tool-call arguments as an object, {} when they are not one."""
    if is_obj(value):
        return value
    if isinstance(value, str) and value.strip() != "":
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return {}
        if is_obj(parsed):
            return parsed
    return {}


def stop_reason_for(finish: object) -> str:
    """OpenAI's finish_reason spelled Anthropic's way."""
    if finish == "length":
        return "max_tokens"
    if finish in ("tool_calls", "function_call"):
        return "tool_use"
    if finish == "content_filter":
        return "refusal"
    return "end_turn"


# ---------------------------------------------------------------- wire translation

def anthropic_request_to_gateway(body: Json, model: str) -> Json:
    """An Anthropic Messages request as a Chat Completions body naming `model`.

    The system prompt becomes a system message, image blocks become image_url parts,
    tool_use blocks become tool_calls, tool_result blocks become tool messages, and the
    output cap, temperature, top_p, stop sequences and streaming pass through.
    """
    messages: list[Json] = []
    system_text = text_in(body.get("system"))
    if system_text != "":
        messages.append({"role": "system", "content": system_text})
    history = body.get("messages") if isinstance(body.get("messages"), list) else []
    for raw in history:
        if not is_obj(raw):
            continue
        from_assistant = raw.get("role") == "assistant"
        content = raw.get("content")
        if isinstance(content, str):
            messages.append({"role": "assistant" if from_assistant else "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        if from_assistant:
            calls: list[Json] = []
            for block in content:
                if not is_obj(block) or block.get("type") != "tool_use":
                    continue
                calls.append(
                    {
                        "id": block["id"] if isinstance(block.get("id"), str) else "",
                        "type": "function",
                        "function": {
                            "name": block["name"] if isinstance(block.get("name"), str) else "",
                            "arguments": json.dumps(
                                block["input"] if is_obj(block.get("input")) else {},
                                separators=(",", ":"),
                                ensure_ascii=False,
                            ),
                        },
                    }
                )
            spoken = text_in(content)
            message: Json = {"role": "assistant", "content": spoken if spoken != "" else None}
            if calls:
                message["tool_calls"] = calls
            messages.append(message)
            continue
        parts: list[Json] = []
        for block in content:
            if not is_obj(block):
                continue
            if block.get("type") == "tool_result":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"] if isinstance(block.get("tool_use_id"), str) else "",
                        "content": tool_result_text(block.get("content")),
                    }
                )
                continue
            part = content_part(block)
            if part is not None:
                parts.append(part)
        if parts:
            messages.append({"role": "user", "content": parts})

    out: Json = {"model": model, "messages": messages}
    if is_int_float(body.get("max_tokens")):
        out["max_tokens"] = body["max_tokens"]
    if is_int_float(body.get("temperature")):
        out["temperature"] = body["temperature"]
    if is_int_float(body.get("top_p")):
        out["top_p"] = body["top_p"]
    if isinstance(body.get("stop_sequences"), list) and len(body["stop_sequences"]) > 0:
        out["stop"] = body["stop_sequences"]
    if isinstance(body.get("tools"), list):
        tools: list[Json] = []
        for raw_tool in body["tools"]:
            if not is_obj(raw_tool) or not isinstance(raw_tool.get("name"), str):
                continue
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": raw_tool["name"],
                        "description": as_str(raw_tool.get("description")),
                        "parameters": raw_tool["input_schema"] if is_obj(raw_tool.get("input_schema")) else {"type": "object", "properties": {}},
                    },
                }
            )
        if tools:
            out["tools"] = tools
    if is_obj(body.get("tool_choice")):
        choice = body["tool_choice"]
        named = as_str(choice.get("name"))
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "tool" and named != "":
            out["tool_choice"] = {"type": "function", "function": {"name": named}}
    if body.get("stream") is True:
        out["stream"] = True
    return out


def gateway_answer_to_anthropic(answer: Json, model: str) -> Json:
    """A Chat Completions answer as an Anthropic Messages object naming `model`."""
    choices = answer.get("choices") if isinstance(answer.get("choices"), list) else []
    choice = choices[0] if choices and is_obj(choices[0]) else {}
    message = choice.get("message") if is_obj(choice.get("message")) else {}
    blocks: list[Json] = []
    spoken = message["content"] if isinstance(message.get("content"), str) else text_in(message.get("content"))
    if spoken != "":
        blocks.append({"type": "text", "text": spoken})
    for call in message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []:
        if not is_obj(call) or not is_obj(call.get("function")):
            continue
        fn = call["function"]
        blocks.append(
            {
                "type": "tool_use",
                "id": call["id"] if isinstance(call.get("id"), str) else "toolu_unknown",
                "name": fn["name"] if isinstance(fn.get("name"), str) else "",
                "input": tool_arguments(fn.get("arguments")),
            }
        )
    if not blocks:
        blocks.append({"type": "text", "text": ""})
    usage = answer.get("usage") if is_obj(answer.get("usage")) else {}
    return {
        "id": f"msg_{answer['id'] if isinstance(answer.get('id'), str) else 'aggregate'}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": stop_reason_for(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {
            "input_tokens": as_num_or(usage.get("prompt_tokens"), 0),
            "output_tokens": as_num_or(usage.get("completion_tokens"), 0),
        },
    }


def gateway_answer_with_model(answer: Json, model: str) -> Json:
    """The answer as it came, its `model` the id this entry dialled: the one truthful claim."""
    return {**answer, "model": model}


# ---------------------------------------------------------------- server-sent events

def sse_frame(event: str | None, data: object) -> str:
    """One server-sent event: an `event:` line when named, the data line, then the blank line."""
    prefix = "" if event is None else f"event: {event}\n"
    return f"{prefix}data: {json.dumps(data, separators=(',', ':'), ensure_ascii=False)}\n\n"


def frame_payload(frame: str) -> str:
    """The frame's data lines joined per the event-stream grammar."""
    return "\n".join(line[5:].removeprefix(" ") for line in frame.split("\n") if line.startswith("data:"))


def frame_renamed(frame: str, model: str) -> str:
    """Keep SSE metadata lines, report the dialled model on the JSON payload."""
    payload = frame_payload(frame)
    if not payload or payload == "[DONE]":
        return frame + "\n\n"
    chunk = json.loads(payload)
    if not is_obj(chunk):
        raise ValueError("gateway stream payload is not an object")
    if chunk.get("error"):
        raise ValueError("gateway stream reported an error")
    metadata = [line for line in frame.split("\n") if not line.startswith("data:")]
    metadata.append("data: " + json.dumps({**chunk, "model": model}, separators=(",", ":"), ensure_ascii=False))
    return "\n".join(metadata) + "\n\n"


def stream_frames(chunks):
    """Decode UTF-8 incrementally, normalize CR and CRLF, and yield whole SSE frames."""
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    buffer = ""
    carry_cr = False
    for chunk in itertools.chain(chunks, [b""]):
        text = decoder.decode(chunk, final=not chunk)
        if carry_cr:
            text = "\r" + text
            carry_cr = False
        if text.endswith("\r") and chunk:
            text = text[:-1]
            carry_cr = True
        buffer += text.replace("\r\n", "\n").replace("\r", "\n")
        while "\n\n" in buffer:
            frame, buffer = buffer.split("\n\n", 1)
            yield frame
    if buffer.strip():
        yield buffer


def openai_stream_renamed(chunks, model: str):
    """The Chat Completions SSE stream as it came, every payload's `model` the dialled id."""
    finished = False
    for frame in stream_frames(chunks):
        if finished:
            break
        yield frame_renamed(frame, model)
        finished = frame_payload(frame) == "[DONE]"
    if not finished:
        raise ValueError("gateway stream ended before its completion marker")


def anthropic_stream_from_gateway(chunks, model: str):
    """A Chat Completions SSE stream re-told as Anthropic's event sequence.

    message_start on the first chunk, content_block_start/delta/stop per text block or
    tool call, message_delta with the stop reason and usage, then message_stop. The
    `model` every event names is the id this entry dialled.
    """
    started = False
    finished = False
    held_stop: str | None = None
    text_block = -1
    next_block = 0
    tool_blocks: dict[object, int] = {}
    tokens_in: int | float = 0
    tokens_out: int | float = 0

    def open_events() -> list[str]:
        nonlocal started
        if started:
            return []
        started = True
        return [
            sse_frame(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_stream",
                        "type": "message",
                        "role": "assistant",
                        "model": model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": tokens_in, "output_tokens": 0},
                    },
                },
            )
        ]

    def close_blocks() -> list[str]:
        nonlocal text_block
        closing: list[str] = []
        if text_block != -1:
            closing.append(sse_frame("content_block_stop", {"type": "content_block_stop", "index": text_block}))
            text_block = -1
        for index in sorted(tool_blocks.values()):
            closing.append(sse_frame("content_block_stop", {"type": "content_block_stop", "index": index}))
        tool_blocks.clear()
        return closing

    def close_events(stop_reason: str) -> list[str]:
        nonlocal finished
        if finished:
            return []
        finished = True
        return [
            *close_blocks(),
            sse_frame(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": {"input_tokens": tokens_in, "output_tokens": tokens_out},
                },
            ),
            sse_frame("message_stop", {"type": "message_stop"}),
        ]

    def absorb(frame: str) -> list[str]:
        nonlocal tokens_in, tokens_out, text_block, next_block, held_stop
        events: list[str] = []
        payload = frame_payload(frame)
        if payload == "[DONE]":
            events.extend(open_events())
            events.extend(close_events(held_stop if held_stop is not None else "end_turn"))
            return events
        try:
            chunk = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            if not payload:
                return []
            raise ValueError("gateway stream payload is not valid JSON")
        if not is_obj(chunk):
            raise ValueError("gateway stream payload is not an object")
        if chunk.get("error"):
            raise ValueError("gateway stream reported an error")
        if is_obj(chunk.get("usage")):
            tokens_in = as_num_or(chunk["usage"].get("prompt_tokens"), 0)
            tokens_out = as_num_or(chunk["usage"].get("completion_tokens"), 0)
        choices = chunk.get("choices") if isinstance(chunk.get("choices"), list) else []
        choice = choices[0] if choices and is_obj(choices[0]) else None
        if choice is None:
            return events
        delta = choice.get("delta") if is_obj(choice.get("delta")) else {}
        events.extend(open_events())
        if isinstance(delta.get("content"), str) and delta["content"] != "":
            if text_block == -1:
                text_block = next_block
                next_block += 1
                events.append(
                    sse_frame(
                        "content_block_start",
                        {"type": "content_block_start", "index": text_block, "content_block": {"type": "text", "text": ""}},
                    )
                )
            events.append(
                sse_frame(
                    "content_block_delta",
                    {"type": "content_block_delta", "index": text_block, "delta": {"type": "text_delta", "text": delta["content"]}},
                )
            )
        for call in delta.get("tool_calls") if isinstance(delta.get("tool_calls"), list) else []:
            if not is_obj(call):
                continue
            upstream_index = call["index"] if is_int_float(call.get("index")) else len(tool_blocks)
            block = tool_blocks.get(upstream_index)
            if block is None:
                if text_block != -1:
                    events.append(sse_frame("content_block_stop", {"type": "content_block_stop", "index": text_block}))
                    text_block = -1
                block = next_block
                next_block += 1
                tool_blocks[upstream_index] = block
                fn = call.get("function") if is_obj(call.get("function")) else {}
                events.append(
                    sse_frame(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": block,
                            "content_block": {
                                "type": "tool_use",
                                "id": call["id"] if isinstance(call.get("id"), str) and call["id"] != "" else f"toolu_{block}",
                                "name": fn["name"] if isinstance(fn.get("name"), str) else "",
                                "input": {},
                            },
                        },
                    )
                )
            fn = call.get("function") if is_obj(call.get("function")) else {}
            if isinstance(fn.get("arguments"), str) and fn["arguments"] != "":
                events.append(
                    sse_frame(
                        "content_block_delta",
                        {"type": "content_block_delta", "index": block, "delta": {"type": "input_json_delta", "partial_json": fn["arguments"]}},
                    )
                )
        # Hold the stop reason until the usage chunk and [DONE] have arrived: OpenAI sends
        # the usage after the finish_reason, and Anthropic's message_delta carries the counts.
        if choice.get("finish_reason") is not None:
            held_stop = stop_reason_for(choice.get("finish_reason"))
        return events

    for frame in stream_frames(chunks):
        for event in absorb(frame):
            yield event
    if not finished:
        raise ValueError("gateway stream ended before its completion marker")


# ---------------------------------------------------------------- routing state

class Router:
    """Deterministic routing: epochs, per-model cooldowns, the trial's call budget.

    A rate limit quarantines a model for the next two request epochs, a server failure
    for one; repeated failures extend the quarantine, capped at eight epochs. No clock
    reading or latency ever influences a choice, so a replay routes identically.
    """

    def __init__(self) -> None:
        self.epoch = 0
        self.calls = 0
        self.blocked_until: dict[str, int] = {}
        self.failure_counts: dict[str, int] = {}
        self.lock = threading.Lock()

    def begin(self) -> int:
        with self.lock:
            self.epoch += 1
            return self.epoch

    def reachable(self, model: str, epoch: int) -> bool:
        with self.lock:
            return epoch > self.blocked_until.get(model, 0)

    def spend(self) -> bool:
        """Reserve one gateway call against the trial's budget; False when it is spent."""
        with self.lock:
            if self.calls >= MAX_CALLS_PER_TRIAL:
                return False
            self.calls += 1
            return True

    def penalize(self, model: str, status: int, epoch: int) -> None:
        if status != 429 and status < 500:
            return
        with self.lock:
            count = self.failure_counts.get(model, 0) + 1
            self.failure_counts[model] = count
            skip = min(8, (2 if status == 429 else 1) * 2 ** min(count - 1, 3))
            self.blocked_until[model] = epoch if status == 503 else epoch + skip

    def reward(self, model: str) -> None:
        with self.lock:
            self.failure_counts.pop(model, None)
            self.blocked_until.pop(model, None)


router = Router()
serve_lock = threading.Lock()


def ranked_candidates(demands: Json) -> list[Json]:
    """The free models that meet `demands`, best rating first, from the live brief."""
    models = BRIEF.get("models")
    if not isinstance(models, list):
        return []
    eligible = [m for m in models if is_obj(m) and satisfies(demands, model_capabilities(m))]
    eligible.sort(key=lambda m: as_num_or(m.get("rating_milli"), 0), reverse=True)
    return eligible


def declared_demands() -> Json:
    """The arriving prompt's demands: the brief's row by arrival order."""
    global arrival_index
    with arrival_lock:
        index = arrival_index
        arrival_index += 1
    rows = BRIEF["prompts"]
    if index < len(rows):
        return dict(rows[index]["requires"])
    return {}


def body_demands(body: Json) -> Json:
    """Demands read off a request body, for prompts outside the declared workload."""
    wanted: Json = {}
    history = body.get("messages") if isinstance(body.get("messages"), list) else []
    for message in history:
        if not is_obj(message) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if is_obj(block) and block.get("type") in ("image", "image_url"):
                wanted["vision"] = True
    response_format = body.get("response_format")
    response_type = response_format.get("type") if is_obj(response_format) else None
    if response_type in ("json_object", "json_schema"):
        wanted["json"] = True
    if isinstance(body.get("tools"), list) and len(body["tools"]) > 0:
        wanted["tools"] = True
    return wanted


def demands_for(body: Json) -> Json:
    """The prompt's demands: the brief's row, strengthened by what the body itself carries."""
    demands = declared_demands()
    for key, value in body_demands(body).items():
        if value:
            demands[key] = value
    return demands


def refusal_row(status: int, text: str, model: str | None) -> Json:
    """One observed HTTP refusal for the x-makefun-upstream-failures header."""
    code = "upstream_http_error"
    try:
        value = json.loads(text)
        envelope = value.get("error", {}) if isinstance(value, dict) else {}
        if isinstance(envelope, dict):
            reported = envelope.get("code") or envelope.get("type")
            if isinstance(reported, (str, int)):
                code = str(reported)[:128]
    except (ValueError, TypeError):
        pass
    return {"status": status, "code": code, "model": model}


def gateway_chat_url() -> str:
    """The gateway's Chat Completions route; OPENAI_BASE_URL may or may not end in /v1."""
    if GATEWAY_ROOT == "":
        return ""
    return f"{GATEWAY_ROOT}/chat/completions" if GATEWAY_ROOT.endswith("/v1") else f"{GATEWAY_ROOT}/v1/chat/completions"


class Upstream:
    """One gateway answer: its status and a body readable whole or as arriving chunks."""

    def __init__(self, status: int, response: object) -> None:
        self.status = status
        self._response = response

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def chunks(self):
        """The body's bytes as they arrive, never buffered whole (SSE must stream)."""
        reader = getattr(self._response, "read1", None) or self._response.read
        while True:
            chunk = reader(65536)
            if not chunk:
                break
            yield chunk

    def text(self) -> str:
        try:
            return self._response.read().decode("utf-8", "replace")
        finally:
            self.close()

    def close(self) -> None:
        try:
            self._response.close()
        except Exception:
            pass


def dial(model: str, body: Json) -> Upstream:
    """POST one Chat Completions request to the gateway, naming the model dialled."""
    url = gateway_chat_url()
    if url == "":
        raise RuntimeError("OPENAI_BASE_URL is not set")
    payload = json.dumps({**body, "model": model}, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    headers = {"content-type": "application/json", "authorization": f"Bearer {GATEWAY_KEY}"}
    if body.get("stream") is True:
        headers["accept"] = "text/event-stream"
    request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        response = urllib.request.urlopen(request, timeout=GATEWAY_TIMEOUT_SECONDS)
    except urllib.error.HTTPError as err:
        return Upstream(err.code, err)
    return Upstream(response.status, response)


def outgoing_body(body: Json, wire: str, model: str) -> Json:
    """The gateway body for one client request: translated, capped, and usage-tracked."""
    out = anthropic_request_to_gateway(body, model) if wire == "anthropic" else dict(body)
    cap_key = "max_completion_tokens" if "max_completion_tokens" in out else "max_tokens"
    cap = out.get(cap_key, 1024)
    if not isinstance(cap, int) or isinstance(cap, bool) or not 1 <= cap <= 8192:
        raise ValueError("output token cap must be an integer between 1 and 8192")
    out[cap_key] = cap
    out["model"] = model
    if out.get("stream") is True:
        options = out.get("stream_options")
        if options is not None and not is_obj(options):
            raise ValueError("stream_options must be an object")
        out["stream_options"] = {**(options or {}), "include_usage": True}
    return out


# ---------------------------------------------------------------- the HTTP server

class Handler(BaseHTTPRequestHandler):
    """The entry's one server: the models route, the two completion wires, nothing else."""

    protocol_version = "HTTP/1.1"
    server_version = "aggregator-py/2"
    sys_version = ""

    def log_message(self, format: str, *args: object) -> None:  # the base class's name
        pass

    def do_GET(self) -> None:  # the base class's name
        self.dispatch()

    def do_POST(self) -> None:  # the base class's name
        self.dispatch()

    # -- plumbing -----------------------------------------------------------------------------------

    def dispatch(self) -> None:
        self.headers_sent = False
        self.observed_refusals: list[Json] = []
        path = urlsplit(self.path).path
        try:
            if self.command == "GET" and path == "/v1/models":
                self.models()
                return
            if self.command == "POST" and path == "/v1/chat/completions":
                self.chat()
                return
            if self.command == "POST" and path == "/v1/messages":
                self.messages()
                return
            wire = "anthropic" if path == "/v1/messages" else "openai"
            self.send_error_body(wire, 404, "not_found", f"no route {self.command} {path}")
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception:
            log(traceback.format_exc())
            if not self.headers_sent:
                wire = "anthropic" if urlsplit(self.path).path.startswith("/v1/messages") else "openai"
                try:
                    self.send_error_body(wire, 500, "internal_error", "the server failed to answer")
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True

    def send_json(self, status: int, body: object) -> None:
        text = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
        raw = text.encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.refusal_header()
        self.headers_sent = True
        self.end_headers()
        self.wfile.write(raw)

    def send_error_body(self, wire: str, status: int, kind: str, message: str) -> None:
        if wire == "anthropic":
            self.send_json(status, anthropic_error_body(kind, message))
        else:
            self.send_json(status, openai_error_body(message, kind, None))

    def refusal_header(self) -> None:
        if self.observed_refusals:
            value = json.dumps(self.observed_refusals[:MAX_TRACKED_FAILURES], separators=(",", ":"), ensure_ascii=True)
            if len(value.encode("ascii")) <= 8192:
                self.send_header("x-makefun-upstream-failures", value)

    def stream_headers(self) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.refusal_header()
        # Python's http.server does not chunk a response without a length; the close is
        # what ends the stream for the client.
        self.send_header("connection", "close")
        self.headers_sent = True
        self.end_headers()
        self.close_connection = True

    def write_event(self, text: str) -> None:
        self.wfile.write(text.encode("utf-8"))
        self.wfile.flush()

    def read_json(self) -> Json | None:
        """One client request body, or None when it is not one JSON object or is too large."""
        if self.headers.get("transfer-encoding") is not None:
            self.close_connection = True
            return None
        length = self.headers.get("content-length")
        if length is None:
            self.close_connection = True
            return None
        try:
            size = int(length)
        except ValueError:
            self.close_connection = True
            return None
        if size < 0 or size > MAX_BODY_BYTES:
            self.close_connection = True
            return None
        raw = self.rfile.read(size) if size > 0 else b""
        if not raw:
            return None
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return None
        return value if is_obj(value) else None

    def give_up(self, wire: str, status: int, text: str) -> None:
        """A refused or broken gateway call as a 502 in the client's wire: the prompt is lost."""
        suffix = "" if text == "" else f": {text[:500]}"
        if wire == "anthropic":
            self.send_error_body(wire, 502, "api_error", f"the gateway answered HTTP {status}{suffix}")
        else:
            self.send_error_body(wire, 502, "upstream_error", f"the gateway answered HTTP {status}{suffix}")

    # -- the routes ---------------------------------------------------------------------------------

    def models(self) -> None:
        """GET /v1/models, in the shape of the auth header the client sent."""
        if self.headers.get("x-api-key") is not None:
            self.send_json(200, anthropic_listing(POOL))
        else:
            self.send_json(200, openai_listing(POOL))

    def chat(self) -> None:
        with serve_lock:
            self.complete("openai")

    def messages(self) -> None:
        with serve_lock:
            self.complete("anthropic")

    def complete(self, wire: str) -> None:
        """One prompt answered: rank, dial, fail over, and answer in the client's wire.

        Retries happen only before client headers are sent; once streaming has begun an
        upstream failure is surfaced as an SSE error event, never a second answer.
        """
        body = self.read_json()
        if body is None:
            self.send_error_body(wire, 400, "invalid_request_error", "the request body is not one JSON object")
            return
        try:
            outgoing_body(body, wire, "auto")
        except (ValueError, TypeError) as err:
            self.send_error_body(wire, 400, "invalid_request_error", str(err))
            return
        epoch = router.begin()
        candidates = ranked_candidates(demands_for(body))
        if not candidates:
            kind = "overloaded_error" if wire == "anthropic" else "no_free_model"
            self.send_error_body(wire, 503, kind, "no free model in the brief meets this request")
            return
        last_status, last_text = 502, "all eligible models failed or are cooling down"
        streaming = body.get("stream") is True
        for route in candidates:
            model_id = route["id"]
            if not router.reachable(model_id, epoch):
                continue
            if not router.spend():
                last_status, last_text = 429, "trial call budget exhausted"
                break
            upstream = None
            try:
                request = outgoing_body(body, wire, model_id)
                upstream = dial(model_id, request)
                if not upstream.ok:
                    last_status = upstream.status
                    router.penalize(model_id, upstream.status, epoch)
                    last_text = upstream.text()
                    self.observed_refusals.append(refusal_row(last_status, last_text, model_id))
                    continue
                if streaming:
                    # Validate the first real data frame before committing HTTP 200.
                    frames = stream_frames(upstream.chunks())
                    first = None
                    for frame in frames:
                        payload = frame_payload(frame)
                        if not payload:
                            continue
                        if payload == "[DONE]":
                            raise ValueError("gateway returned an empty stream")
                        parsed = json.loads(payload)
                        if not is_obj(parsed) or parsed.get("error") or not isinstance(parsed.get("choices"), list):
                            raise ValueError("gateway returned an invalid stream")
                        first = frame
                        break
                    if first is None:
                        raise ValueError("gateway returned an empty stream")
                    chunks = ((frame + "\n\n").encode("utf-8") for frame in itertools.chain([first], frames))
                    if wire == "anthropic":
                        events = anthropic_stream_from_gateway(chunks, model_id)
                    else:
                        events = openai_stream_renamed(chunks, model_id)
                    self.stream_headers()
                    for event in events:
                        self.write_event(event)
                else:
                    parsed = json.loads(upstream.text())
                    if (
                        not is_obj(parsed)
                        or parsed.get("error")
                        or not isinstance(parsed.get("choices"), list)
                        or not parsed["choices"]
                        or not is_obj(parsed["choices"][0])
                        or not is_obj(parsed["choices"][0].get("message"))
                    ):
                        raise ValueError("gateway returned an invalid completion")
                    answer = (
                        gateway_answer_to_anthropic(parsed, model_id)
                        if wire == "anthropic"
                        else gateway_answer_with_model(parsed, model_id)
                    )
                    self.send_json(200, answer)
                router.reward(model_id)
                return
            except (BrokenPipeError, ConnectionResetError):
                raise
            except Exception as err:
                last_status, last_text = 502, str(err)
                router.penalize(model_id, 502, epoch)
                if self.headers_sent:
                    error = (
                        anthropic_error_body("api_error", "upstream stream failed")
                        if wire == "anthropic"
                        else openai_error_body("upstream stream failed", "upstream_error", None)
                    )
                    self.write_event(sse_frame("error" if wire == "anthropic" else None, error))
                    return
            finally:
                if upstream is not None:
                    upstream.close()
        self.give_up(wire, last_status, last_text)


# ---------------------------------------------------------------- register and poll

def register(port: int, report: Json) -> None:
    """POST /register with the port and the one catalog; the env answers with the prompt count."""
    payload = json.dumps(
        {"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": report},
        separators=(",", ":"),
        ensure_ascii=False,
    )
    request = urllib.request.Request(
        f"{ENV_URL}/register",
        data=payload.encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=REGISTER_TIMEOUT_SECONDS) as response:
            status = response.status
            reply_text = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as err:
        raise RuntimeError(f"POST {ENV_URL}/register answered HTTP {err.code}: {err.read().decode('utf-8', 'replace')[:500]}") from err
    if not (200 <= status < 300):
        raise RuntimeError(f"POST {ENV_URL}/register answered HTTP {status}: {reply_text[:500]}")
    try:
        reply = json.loads(reply_text)
    except (json.JSONDecodeError, ValueError):
        reply = None
    if is_obj(reply) and is_int_float(reply.get("requests")):
        log(f"registered: the trial holds {reply['requests']} prompts")


def poll() -> None:
    """Poll GET /work every 100 ms; exit 0 the moment it says done."""
    while True:
        time.sleep(WORK_POLL_SECONDS)
        try:
            with urllib.request.urlopen(f"{ENV_URL}/work", timeout=2) as response:
                if response.status != 200:
                    continue
                work = json.loads(response.read().decode("utf-8", "replace"))
        except Exception:
            continue
        if is_obj(work) and work.get("done") is True:
            sys.stderr.flush()
            os._exit(0)


def stop(*_: object) -> None:
    """SIGTERM and SIGINT end the trial as Node's process.exit(0) does: exit 0, at once."""
    sys.stderr.flush()
    os._exit(0)


def start() -> None:
    if BRIEF["fault"] is not None:
        log(BRIEF["fault"])
    if "--list" in sys.argv[1:]:
        # The same document the register body carries, byte for byte: print it and stop
        # before any server starts. Flush so a buffered stdout cannot lose the tail.
        sys.stdout.write(json.dumps(CATALOG, separators=(",", ":"), ensure_ascii=False) + "\n")
        sys.stdout.flush()
        return
    if ENV_URL == "":
        raise RuntimeError("MAKEFUN_ENV_URL is not set")
    if gateway_chat_url() == "":
        raise RuntimeError("OPENAI_BASE_URL is not set")
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    try:
        port = server.server_address[1]
        if port == PORT_FORBIDDEN:
            raise RuntimeError("the OS picked 8080, which the env owns on one layout: restart the trial")
        log(f"listening on 0.0.0.0:{port} with {len(POOL)} free models")
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        threading.Thread(target=server.serve_forever, name="http-server", daemon=True).start()
        register(port, CATALOG)
        poll()
    finally:
        server.server_close()


def main() -> int:
    try:
        start()
        return 0
    except SystemExit:
        raise
    except BaseException:
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
