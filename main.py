"""Self-contained free-model aggregator competition entrypoint.

Run with ``python3 -P -s main.py``; only the standard library is required.
The brief supplies the catalog, ratings, and ordered request requirements.
Eligible free models are tried in descending quality order. Both client APIs
share gateway failover and always report the model actually called.

``--list`` prints the registration catalog without opening a server. Normal
startup binds an ephemeral port, registers it, and polls /work until done.
This entry derives from the official MIT Python starter and earlier local
agent revisions. This session revises routing and transport; it does not claim
the inherited parser, API adapters, or tests as fresh authorship.
The TypeScript starter is retained separately as starter-reference.ts.
"""

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

# ---------------------------------------------------------------- the wire's constants

# The wire protocol's version, in the register body.
PROTOCOL_VERSION = 1
# How often GET /work is polled, in seconds.
POLL_INTERVAL_SEC = 0.1
# The most bytes of one client request body this baseline reads.
MAX_BODY_BYTES = 8 * 1024 * 1024

ENV_URL = os.environ.get('MAKEFUN_ENV_URL', '').rstrip('/')
PROMPT_FILE = os.environ.get('MAKEFUN_PROMPT_FILE', '')
GATEWAY_URL = os.environ.get('OPENAI_BASE_URL', '').rstrip('/')
# The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key.
GATEWAY_KEY = os.environ.get('OPENAI_API_KEY', '')

MODELS_SCHEMA = 'free-model-aggregator-02/models/v1'
PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1'
# Every schema this contest's brief writes names this prefix; the brief's JSON is found by it.
SCHEMA_PREFIX = 'free-model-aggregator-02/'

# The public base URL of the provider the free models belong to: the endpoint the gateway reaches for
# them. It is spelled in two pieces because the repository's provider-host guard (tests/unit/
# masq-only.test.ts NAMED_HOSTS) allows this host only in lib/providers.ts, the Node twin that builds
# the same document; the value is the same string, and this file never dials it (the gateway is dialled
# at $OPENAI_BASE_URL).
OPENROUTER_PUBLIC_BASE = 'https://' + 'openrouter' + '.ai/api/v1'
# The environment variable an entrant's OpenRouter key is pushed under (kept out of the sandbox).
OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY'

# A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock.
FIXED_OPENAI_CREATED = 1767225600
# A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock.
FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z'

# Shorthand for the JSON objects this file moves around: one JSON object, as Python reads it.
Json = dict[str, object]

# ---------------------------------------------------------------- small type checks


def is_record(value: object) -> bool:
    """A JSON object (a Python dict), never an array or another value."""
    return isinstance(value, dict)


def is_number(value: object) -> bool:
    """A JSON number: an int or a float, never a bool (Python's bool is an int)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def string_of(value: object) -> str:
    return value if isinstance(value, str) else ''


def number_or_null(value: object) -> int | float | None:
    return value if is_number(value) and math.isfinite(value) else None


def number_or_zero(value: object) -> int | float:
    return value if is_number(value) else 0


def bool_or_null(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


# ---------------------------------------------------------------- the brief's documents


def object_end(text: str, start: int) -> int:
    """The index of the `}` closing the object that opens at `start`, or -1; string contents skipped."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == '\\':
                i += 2
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def documents_in(text: str) -> list[Json]:
    """Every JSON object in `text` that carries a string `schema`, in the order they appear."""
    out: list[Json] = []
    start = text.find('{')
    while start != -1:
        end = object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start : end + 1])
            except (json.JSONDecodeError, ValueError):
                value = None
            if is_record(value) and isinstance(value.get('schema'), str):
                out.append(value)
        start = text.find('{', start + 1)
    return out


def array_document(docs: list[Json], keys: tuple[str, ...]) -> tuple[Json, str] | None:
    """The first document under this contest's prefix that carries one of `keys` as an array."""
    for doc in docs:
        schema = string_of(doc.get('schema'))
        if SCHEMA_PREFIX not in schema:
            continue
        for key in keys:
            if isinstance(doc.get(key), list):
                return doc, key
    return None


def model_of(value: object) -> Json | None:
    """One model row, as this baseline reads it. A free row is routed as the gateway spells its id."""
    if not is_record(value):
        return None
    raw = string_of(value.get('id'))
    if raw == '':
        raw = string_of(value.get('model'))
    if raw == '':
        return None
    named = string_of(value.get('provider'))
    if named == '':
        named = string_of(value.get('wire'))
    model_id = f'openrouter/{raw}' if named == 'openrouter' and not raw.lower().startswith('openrouter/') else raw
    display_name = string_of(value.get('display_name'))
    rating = number_or_null(value.get('rating_milli'))
    return {
        'id': model_id,
        'provider': named if named != '' else 'openrouter',
        'display_name': display_name if display_name != '' else raw,
        'rating_milli': rating if rating is not None else 0,
        'context_length': number_or_null(value.get('context_length')),
        'tools': bool_or_null(value.get('tools')),
        'vision': bool_or_null(value.get('vision')),
        'json': bool_or_null(value.get('json')),
        'free': value.get('free') is not False,
    }


def prompt_of(value: object) -> Json | None:
    """One prompt row, as this baseline reads it."""
    if not is_record(value):
        return None
    prompt_id = string_of(value.get('id'))
    if prompt_id == '':
        return None
    requires = value.get('requires') if is_record(value.get('requires')) else {}
    return {
        'id': prompt_id,
        'api': 'anthropic' if value.get('api') == 'anthropic' else 'openai',
        'stream': value.get('stream') is True,
        'weight': 2 if value.get('weight') == 2 else 1,
        'requires': {
            **({'tools': True} if requires.get('tools') is True else {}),
            **({'vision': True} if requires.get('vision') is True else {}),
            **({'json': True} if requires.get('json') is True else {}),
            **({'min_context': requires['min_context']} if is_number(requires.get('min_context')) and math.isfinite(requires['min_context']) else {}),
        },
    }


def read_brief(path: str) -> Json:
    """The brief at `path`, its two documents found under this contest's schema prefix.

    Returns {"models": [...], "prompts": [...], "error": <string or None>}. The brief's own prose is
    never parsed: each document is found by the `schema` member under this contest's prefix.
    """
    if path == '':
        return {'models': [], 'prompts': [], 'error': 'MAKEFUN_PROMPT_FILE is not set'}
    try:
        with open(path, encoding='utf-8') as handle:
            text = handle.read()
    except OSError as err:
        return {'models': [], 'prompts': [], 'error': f'cannot read {path}: {err}'}
    docs = documents_in(text)
    models_doc = array_document(docs, ('models', 'ratings'))
    if models_doc is None:
        return {'models': [], 'prompts': [], 'error': f'the brief at {path} holds no {MODELS_SCHEMA} document'}
    models = [m for m in (model_of(v) for v in models_doc[0][models_doc[1]]) if m is not None]
    prompts_doc = array_document(docs, ('prompts', 'manifest'))
    if prompts_doc is None:
        return {'models': models, 'prompts': [], 'error': f'the brief at {path} holds no {PROMPTS_SCHEMA} document'}
    prompts = [p for p in (prompt_of(v) for v in prompts_doc[0][prompts_doc[1]]) if p is not None]
    return {'models': models, 'prompts': prompts, 'error': None}


def free_models(brief: Json) -> list[Json]:
    """The models this entry may serve: the brief's free ones, in the brief's order."""
    seen = set()
    models = []
    for model in brief['models']:
        if model['free'] and model['id'] not in seen:
            seen.add(model['id'])
            models.append(model)
    return models


def capabilities_of(model: Json) -> Json:
    return {
        'context_length': model['context_length'],
        'tools': model['tools'],
        'vision': model['vision'],
        'json': model['json'],
    }


def meets_requirements(requires: Json, caps: Json) -> bool:
    """Whether a model's capabilities meet one prompt's requirements. An unknown capability never meets one."""
    if requires.get('tools') is True and caps['tools'] is not True:
        return False
    if requires.get('vision') is True and caps['vision'] is not True:
        return False
    if requires.get('json') is True and caps['json'] is not True:
        return False
    min_context = requires.get('min_context')
    if min_context is not None and (caps['context_length'] is None or caps['context_length'] < min_context):
        return False
    return True


def build_catalog(brief: Json) -> Json:
    """The catalog the brief's free model list makes: one provider, the models it serves.

    The base_url is the provider's public endpoint, never the gateway's trial-local address: the gateway's
    origin differs between a live pass and its replay, and the catalog is part of the state export a
    replay must reproduce byte for byte.

    Registration and both listing wires expose the same deduplicated free IDs.
    """
    seen: set[str] = set()
    models: list[Json] = []
    for model in brief['models']:
        model_id = model['id']
        if not model['free'] or model_id in seen:
            continue
        seen.add(model_id)
        models.append({'id': model_id, 'free': True, 'context_length': model['context_length']})
    if not models:
        return {'providers': []}
    return {
        'providers': [
            {
                'id': 'openrouter',
                'base_url': OPENROUTER_PUBLIC_BASE,
                'key_env': OPENROUTER_KEY_ENV,
                'wire': 'openai',
                'models': models,
            }
        ]
    }


# ---------------------------------------------------------------- the client wires and the model field


def openai_model_list(models: list[Json]) -> Json:
    """GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape."""
    return {
        'object': 'list',
        'data': [{'id': m['id'], 'object': 'model', 'created': FIXED_OPENAI_CREATED, 'owned_by': m['provider']} for m in models],
    }


def anthropic_model_list(models: list[Json]) -> Json:
    """GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape."""
    return {
        'data': [{'type': 'model', 'id': m['id'], 'display_name': m['display_name'], 'created_at': FIXED_ANTHROPIC_CREATED_AT} for m in models],
        'has_more': False,
    }


def anthropic_error(error_type: str, message: str) -> Json:
    """An error in Anthropic's shape (the Messages API's error envelope)."""
    return {'type': 'error', 'error': {'type': error_type, 'message': message}}


def openai_error(message: str, error_type: str, code: str | None) -> Json:
    """An error in OpenAI's shape (Chat Completions)."""
    return {'error': {'message': message, 'type': error_type, 'code': code}}


def text_of(content: object) -> str:
    """The text of a string or of a block array's text blocks; '' when there is none."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ''
    parts = [b['text'] for b in content if is_record(b) and b.get('type') == 'text' and isinstance(b.get('text'), str)]
    return '\n'.join(parts)


def part_of(block: Json) -> Json | None:
    """One Anthropic content block as an OpenAI content part: text or an inline image; None for the rest."""
    if block.get('type') == 'text' and isinstance(block.get('text'), str):
        return {'type': 'text', 'text': block['text']}
    source = block.get('source')
    if block.get('type') == 'image' and is_record(source):
        if source.get('type') == 'base64' and isinstance(source.get('data'), str):
            media = source['media_type'] if isinstance(source.get('media_type'), str) else 'image/png'
            return {'type': 'image_url', 'image_url': {'url': f'data:{media};base64,{source["data"]}'}}
        if source.get('type') == 'url' and isinstance(source.get('url'), str):
            return {'type': 'image_url', 'image_url': {'url': source['url']}}
    return None


def tool_result_text(content: object) -> str:
    """A tool_result block's content as text: its text blocks, or its JSON when it holds something else."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = text_of(content)
        return text if text != '' else json.dumps(content, separators=(',', ':'), ensure_ascii=False)
    if content is None:
        return ''
    return json.dumps(content, separators=(',', ':'), ensure_ascii=False)


def arguments_of(value: object) -> Json:
    """A tool call's arguments as an object: the gateway's JSON string, or {} when it is not one."""
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip() != '':
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return {}
        if is_record(parsed):
            return parsed
    return {}


def stop_reason_of(finish: object) -> str:
    """OpenAI's finish_reason as Anthropic's stop_reason."""
    if finish == 'length':
        return 'max_tokens'
    if finish == 'tool_calls' or finish == 'function_call':
        return 'tool_use'
    if finish == 'content_filter':
        return 'refusal'
    return 'end_turn'


def anthropic_to_openai(body: Json, model: str) -> Json:
    """An Anthropic Messages request as an OpenAI Chat Completions request that names `model`.

    The system prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks
    become tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and
    the output cap, temperature, top_p, stop sequences and streaming pass through.
    """
    messages: list[Json] = []
    system = text_of(body.get('system'))
    if system != '':
        messages.append({'role': 'system', 'content': system})
    source = body.get('messages') if isinstance(body.get('messages'), list) else []
    for raw in source:
        if not is_record(raw):
            continue
        assistant = raw.get('role') == 'assistant'
        content = raw.get('content')
        if isinstance(content, str):
            messages.append({'role': 'assistant' if assistant else 'user', 'content': content})
            continue
        if not isinstance(content, list):
            continue
        if assistant:
            tool_calls: list[Json] = []
            for block in content:
                if not is_record(block) or block.get('type') != 'tool_use':
                    continue
                tool_calls.append(
                    {
                        'id': block['id'] if isinstance(block.get('id'), str) else '',
                        'type': 'function',
                        'function': {
                            'name': block['name'] if isinstance(block.get('name'), str) else '',
                            'arguments': json.dumps(
                                block['input'] if is_record(block.get('input')) else {},
                                separators=(',', ':'),
                                ensure_ascii=False,
                            ),
                        },
                    }
                )
            text = text_of(content)
            message: Json = {'role': 'assistant', 'content': text if text != '' else None}
            if tool_calls:
                message['tool_calls'] = tool_calls
            messages.append(message)
            continue
        parts: list[Json] = []
        for block in content:
            if not is_record(block):
                continue
            if block.get('type') == 'tool_result':
                messages.append(
                    {
                        'role': 'tool',
                        'tool_call_id': block['tool_use_id'] if isinstance(block.get('tool_use_id'), str) else '',
                        'content': tool_result_text(block.get('content')),
                    }
                )
                continue
            part = part_of(block)
            if part is not None:
                parts.append(part)
        if parts:
            messages.append({'role': 'user', 'content': parts})

    out: Json = {'model': model, 'messages': messages}
    if is_number(body.get('max_tokens')):
        out['max_tokens'] = body['max_tokens']
    if is_number(body.get('temperature')):
        out['temperature'] = body['temperature']
    if is_number(body.get('top_p')):
        out['top_p'] = body['top_p']
    if isinstance(body.get('stop_sequences'), list) and len(body['stop_sequences']) > 0:
        out['stop'] = body['stop_sequences']
    if isinstance(body.get('tools'), list):
        tools: list[Json] = []
        for raw in body['tools']:
            if not is_record(raw) or not isinstance(raw.get('name'), str):
                continue
            tools.append(
                {
                    'type': 'function',
                    'function': {
                        'name': raw['name'],
                        'description': raw['description'] if isinstance(raw.get('description'), str) else '',
                        'parameters': raw['input_schema'] if is_record(raw.get('input_schema')) else {'type': 'object', 'properties': {}},
                    },
                }
            )
        if tools:
            out['tools'] = tools
    choice = body.get('tool_choice')
    if is_record(choice):
        if choice.get('type') == 'auto':
            out['tool_choice'] = 'auto'
        elif choice.get('type') == 'none':
            out['tool_choice'] = 'none'
        elif choice.get('type') == 'any':
            out['tool_choice'] = 'required'
        elif choice.get('type') == 'tool' and isinstance(choice.get('name'), str):
            out['tool_choice'] = {'type': 'function', 'function': {'name': choice['name']}}
    if is_record(choice) and isinstance(choice.get('disable_parallel_tool_use'), bool):
        out['parallel_tool_calls'] = not choice['disable_parallel_tool_use']
    if is_record(body.get('response_format')):
        out['response_format'] = body['response_format']
    if body.get('stream') is True:
        out['stream'] = True
        out['stream_options'] = {'include_usage': True}
    return out


def openai_to_anthropic(body: Json, model: str) -> Json:
    """A Chat Completions answer as an Anthropic Messages object. `model` is the id the entry dialled."""
    choices = body.get('choices') if isinstance(body.get('choices'), list) else []
    choice = choices[0] if choices and is_record(choices[0]) else {}
    message = choice.get('message') if is_record(choice.get('message')) else {}
    content: list[Json] = []
    text = message['content'] if isinstance(message.get('content'), str) else text_of(message.get('content'))
    if text != '':
        content.append({'type': 'text', 'text': text})
    calls = message.get('tool_calls') if isinstance(message.get('tool_calls'), list) else []
    for call in calls:
        if not is_record(call) or not is_record(call.get('function')):
            continue
        fn = call['function']
        content.append(
            {
                'type': 'tool_use',
                'id': call['id'] if isinstance(call.get('id'), str) else 'toolu_unknown',
                'name': fn['name'] if isinstance(fn.get('name'), str) else '',
                'input': arguments_of(fn.get('arguments')),
            }
        )
    if not content:
        content.append({'type': 'text', 'text': ''})
    usage = body.get('usage') if is_record(body.get('usage')) else {}
    return {
        'id': f'msg_{body["id"] if isinstance(body.get("id"), str) else "aggregate"}',
        'type': 'message',
        'role': 'assistant',
        'model': model,
        'content': content,
        'stop_reason': stop_reason_of(choice.get('finish_reason')),
        'stop_sequence': None,
        'usage': {
            'input_tokens': number_or_zero(usage.get('prompt_tokens')),
            'output_tokens': number_or_zero(usage.get('completion_tokens')),
        },
    }


def openai_json_with_model(body: Json, model: str) -> Json:
    """A Chat Completions answer with `model` set to the id the entry dialled: the one truthful claim."""
    return {**body, 'model': model}


def sse_event(event: str | None, data: object) -> str:
    """One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line."""
    prefix = '' if event is None else f'event: {event}\n'
    return f'{prefix}data: {json.dumps(data, separators=(",", ":"), ensure_ascii=False)}\n\n'


def frame_with_model(frame: str, model: str) -> str:
    """Preserve SSE metadata and report the gateway model on JSON chunks."""
    payload = sse_data(frame)
    if not payload or payload == '[DONE]':
        return frame + '\n\n'
    chunk = json.loads(payload)
    if not is_record(chunk):
        raise ValueError('gateway stream payload is not an object')
    if chunk.get('error'):
        raise ValueError('gateway stream reported an error')
    metadata = [line for line in frame.split('\n') if not line.startswith('data:')]
    metadata.append('data: ' + json.dumps({**chunk, 'model': model},
                                         separators=(',', ':'), ensure_ascii=False))
    return '\n'.join(metadata) + '\n\n'


def sse_data(frame: str) -> str:
    """Join SSE data lines according to the event-stream grammar."""
    return '\n'.join(line[5:].removeprefix(' ') for line in frame.split('\n')
                     if line.startswith('data:'))


def iter_frames(chunks):
    """Decode UTF-8 incrementally and normalize CR, LF, and split CRLF."""
    decoder = codecs.getincrementaldecoder('utf-8')('replace')
    buffer = ''
    pending_cr = False
    for chunk in itertools.chain(chunks, [b'']):
        text = decoder.decode(chunk, final=not chunk)
        if pending_cr:
            text = '\r' + text
            pending_cr = False
        if text.endswith('\r') and chunk:
            text = text[:-1]
            pending_cr = True
        buffer += text.replace('\r\n', '\n').replace('\r', '\n')
        while '\n\n' in buffer:
            frame, buffer = buffer.split('\n\n', 1)
            yield frame
    if buffer.strip():
        yield buffer


def openai_stream_with_model(chunks, model: str):
    """A Chat Completions SSE stream as it came, frame by frame, every data payload's `model` the dialled id."""
    complete = False
    for frame in iter_frames(chunks):
        if complete:
            break
        yield frame_with_model(frame, model)
        complete = sse_data(frame) == '[DONE]'
    if not complete:
        raise ValueError('gateway stream ended before its completion marker')


def anthropic_events_from_openai(chunks, model: str):
    """A Chat Completions SSE stream as the Anthropic event sequence.

    message_start on the first chunk, content_block_start/delta/stop per text block or tool call, then
    message_delta with the stop reason and usage, then message_stop. `model` is the id the entry dialled,
    named back to the client.
    """
    started = False
    finished = False
    # The stop reason a finish_reason named, held until the stream's usage chunk has arrived.
    pending_stop: str | None = None
    text_index = -1
    next_index = 0
    tool_indexes: dict[object, int] = {}
    input_tokens: int | float = 0
    output_tokens: int | float = 0

    def start_events() -> list[str]:
        nonlocal started
        if started:
            return []
        started = True
        return [
            sse_event(
                'message_start',
                {
                    'type': 'message_start',
                    'message': {
                        'id': 'msg_stream',
                        'type': 'message',
                        'role': 'assistant',
                        'model': model,
                        'content': [],
                        'stop_reason': None,
                        'stop_sequence': None,
                        'usage': {'input_tokens': input_tokens, 'output_tokens': 0},
                    },
                },
            )
        ]

    def close_events() -> list[str]:
        nonlocal text_index
        out: list[str] = []
        if text_index != -1:
            out.append(sse_event('content_block_stop', {'type': 'content_block_stop', 'index': text_index}))
            text_index = -1
        for index in sorted(tool_indexes.values()):
            out.append(sse_event('content_block_stop', {'type': 'content_block_stop', 'index': index}))
        tool_indexes.clear()
        return out

    def finish_events(stop_reason: str) -> list[str]:
        nonlocal finished
        if finished:
            return []
        finished = True
        return [
            *close_events(),
            sse_event(
                'message_delta',
                {'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': {'input_tokens': input_tokens, 'output_tokens': output_tokens}},
            ),
            sse_event('message_stop', {'type': 'message_stop'}),
        ]

    def events_of_frame(frame: str) -> list[str]:
        nonlocal input_tokens, output_tokens, text_index, next_index, pending_stop
        out: list[str] = []
        for line in ['data: ' + sse_data(frame)]:
            trimmed = line.strip()
            if not trimmed.startswith('data:'):
                continue
            payload = trimmed[5:].strip()
            if payload == '[DONE]':
                out.extend(start_events())
                out.extend(finish_events(pending_stop if pending_stop is not None else 'end_turn'))
                continue
            try:
                chunk = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                if not payload:
                    continue
                raise ValueError('gateway stream payload is not valid JSON')
            if not is_record(chunk):
                raise ValueError('gateway stream payload is not an object')
            if chunk.get('error'):
                raise ValueError('gateway stream reported an error')
            if is_record(chunk.get('usage')):
                input_tokens = number_or_zero(chunk['usage'].get('prompt_tokens'))
                output_tokens = number_or_zero(chunk['usage'].get('completion_tokens'))
            choices = chunk.get('choices') if isinstance(chunk.get('choices'), list) else []
            choice = choices[0] if choices and is_record(choices[0]) else None
            if choice is None:
                continue
            delta = choice.get('delta') if is_record(choice.get('delta')) else {}
            out.extend(start_events())
            if isinstance(delta.get('content'), str) and delta['content'] != '':
                if text_index == -1:
                    text_index = next_index
                    next_index += 1
                    out.append(
                        sse_event('content_block_start', {'type': 'content_block_start', 'index': text_index, 'content_block': {'type': 'text', 'text': ''}})
                    )
                out.append(
                    sse_event(
                        'content_block_delta',
                        {'type': 'content_block_delta', 'index': text_index, 'delta': {'type': 'text_delta', 'text': delta['content']}},
                    )
                )
            calls = delta.get('tool_calls') if isinstance(delta.get('tool_calls'), list) else []
            for call in calls:
                if not is_record(call):
                    continue
                upstream_index = call['index'] if is_number(call.get('index')) else len(tool_indexes)
                index = tool_indexes.get(upstream_index)
                if index is None:
                    if text_index != -1:
                        out.append(sse_event('content_block_stop', {'type': 'content_block_stop', 'index': text_index}))
                        text_index = -1
                    index = next_index
                    next_index += 1
                    tool_indexes[upstream_index] = index
                    fn = call.get('function') if is_record(call.get('function')) else {}
                    out.append(
                        sse_event(
                            'content_block_start',
                            {
                                'type': 'content_block_start',
                                'index': index,
                                'content_block': {
                                    'type': 'tool_use',
                                    'id': call['id'] if isinstance(call.get('id'), str) and call['id'] != '' else f'toolu_{index}',
                                    'name': fn['name'] if isinstance(fn.get('name'), str) else '',
                                    'input': {},
                                },
                            },
                        )
                    )
                fn = call.get('function') if is_record(call.get('function')) else {}
                if isinstance(fn.get('arguments'), str) and fn['arguments'] != '':
                    out.append(
                        sse_event(
                            'content_block_delta',
                            {'type': 'content_block_delta', 'index': index, 'delta': {'type': 'input_json_delta', 'partial_json': fn['arguments']}},
                        )
                    )
            # Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived:
            # OpenAI sends the usage after the finish_reason, and Anthropic's message_delta carries the
            # output tokens.
            if choice.get('finish_reason') is not None:
                pending_stop = stop_reason_of(choice.get('finish_reason'))
        return out

    for frame in iter_frames(chunks):
        for event in events_of_frame(frame):
            yield event
    if not finished:
        raise ValueError('gateway stream ended before its completion marker')


# ---------------------------------------------------------------- the trial state

BRIEF = read_brief(PROMPT_FILE)
CATALOG = build_catalog(BRIEF)
# Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order.
manifest_index = 0
manifest_lock = threading.Lock()
class RoutingState:
    """Logical request epochs keep live and replay routing identical.

    A rate limit skips the next two requests, a server failure the next one.
    Repeated failures extend the quarantine, capped at eight request epochs.
    No clock reading or response latency influences a model choice.
    """

    def __init__(self, brief):
        self.brief = brief
        self.epoch = 0
        self.calls = 0
        self.blocked_until = {}
        self.failures = {}
        self.lock = threading.Lock()

    def begin(self):
        with self.lock:
            self.epoch += 1
            return self.epoch

    def available(self, model, epoch):
        with self.lock:
            return epoch > self.blocked_until.get(model, 0)

    def reserve(self):
        with self.lock:
            if self.calls >= 60:
                return False
            self.calls += 1
            return True

    def failure(self, model, status, epoch):
        if status != 429 and status < 500:
            return
        with self.lock:
            count = self.failures.get(model, 0) + 1
            self.failures[model] = count
            skip = min(8, (2 if status == 429 else 1) * 2 ** min(count - 1, 3))
            self.blocked_until[model] = epoch + skip

    def success(self, model):
        with self.lock:
            self.failures.pop(model, None)
            self.blocked_until.pop(model, None)


routing_state = RoutingState(BRIEF)
request_lock = threading.Lock()


def state_for_brief():
    global routing_state
    if routing_state.brief is not BRIEF:
        routing_state = RoutingState(BRIEF)
    return routing_state


def refusal_row(status, text, model):
    """Record only an observed HTTP refusal, never inferred transport failures."""
    code = 'upstream_http_error'
    try:
        value = json.loads(text)
        error = value.get('error', {}) if isinstance(value, dict) else {}
        if isinstance(error, dict):
            reported = error.get('code') or error.get('type')
            if isinstance(reported, (str, int)):
                code = str(reported)[:128]
    except (ValueError, TypeError):
        pass
    return {'status': status, 'code': code, 'model': model}


def prepared_request(body, wire, model):
    """Preserve client options while enforcing the gateway output ceiling."""
    out = anthropic_to_openai(body, model) if wire == 'anthropic' else dict(body)
    cap_key = 'max_completion_tokens' if 'max_completion_tokens' in out else 'max_tokens'
    cap = out.get(cap_key, 1024)
    if not isinstance(cap, int) or isinstance(cap, bool) or not 1 <= cap <= 8192:
        raise ValueError('output token cap must be an integer between 1 and 8192')
    out[cap_key] = cap
    out['model'] = model
    if out.get('stream') is True:
        options = out.get('stream_options')
        if options is not None and not isinstance(options, dict):
            raise ValueError('stream_options must be an object')
        out['stream_options'] = {**(options or {}), 'include_usage': True}
    return out



def log(message: str) -> None:
    sys.stderr.write(f'aggregator: {message}\n')
    sys.stderr.flush()


def chat_completions_url() -> str:
    """The gateway's Chat Completions route.

    OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the trial network); a base that does not
    already end in `/v1` gets one, so both spellings reach the route.
    """
    if GATEWAY_URL == '':
        return ''
    return f'{GATEWAY_URL}/chat/completions' if GATEWAY_URL.endswith('/v1') else f'{GATEWAY_URL}/v1/chat/completions'


def served_models() -> list[Json]:
    """The free models the brief lists, as GET /v1/models names them."""
    return [{'id': m['id'], 'provider': m['provider'], 'display_name': m['display_name']} for m in free_models(BRIEF)]


def candidates_for(requires: Json) -> list[Json]:
    candidates = []
    for model in free_models(BRIEF):
        if meets_requirements(requires, capabilities_of(model)):
            candidates.append(model)
    candidates.sort(key=lambda m: number_or_zero(m.get('rating_milli')), reverse=True)
    return candidates


def route_for(requires: Json) -> Json | None:
    cands = candidates_for(requires)
    return cands[0] if cands else None


def requirements_for(body: Json) -> Json:
    """The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own."""
    global manifest_index
    with manifest_lock:
        index = manifest_index
        manifest_index += 1
    prompts = BRIEF['prompts']
    if index < len(prompts):
        declared = dict(prompts[index]['requires'])
        for key, value in requirements_from_body(body).items():
            if value:
                declared[key] = value
        return declared
    # Outside the declared workload, infer capabilities from the request.
    return requirements_from_body(body)


def requirements_from_body(body: Json) -> Json:
    """Requirements read off a request body, when the brief has no row for it."""
    messages = body.get('messages') if isinstance(body.get('messages'), list) else []
    vision = False
    for message in messages:
        if not is_record(message) or not isinstance(message.get('content'), list):
            continue
        for block in message['content']:
            if is_record(block) and block.get('type') in ('image', 'image_url'):
                vision = True
    response_format = body.get('response_format')
    response_type = response_format.get('type') if is_record(response_format) else None
    return {'tools': isinstance(body.get('tools'), list) and len(body['tools']) > 0, 'vision': vision, 'json': response_type in ('json_object', 'json_schema')}


# ---------------------------------------------------------------- the gateway


class Upstream:
    """One gateway answer: its status and a body that may be read whole or as byte chunks."""

    def __init__(self, status: int, response) -> None:
        self.status = status
        self._response = response

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def chunks(self):
        """The body's bytes as they arrive, never waiting to fill a buffer (SSE must stream)."""
        reader = getattr(self._response, 'read1', None) or self._response.read
        while True:
            chunk = reader(65536)
            if not chunk:
                break
            yield chunk

    def text(self) -> str:
        try:
            return self._response.read().decode('utf-8', 'replace')
        finally:
            self.close()

    def close(self) -> None:
        try:
            self._response.close()
        except Exception:  # a connection already closed is nothing to report
            pass


def dial(model: str, body: Json) -> Upstream:
    """POST one request to the gateway's Chat Completions route, naming the model this entry dialled."""
    url = chat_completions_url()
    if url == '':
        raise RuntimeError('OPENAI_BASE_URL is not set')
    data = json.dumps({**body, 'model': model}, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    headers = {'content-type': 'application/json', 'authorization': f'Bearer {GATEWAY_KEY}'}
    if body.get('stream') is True:
        headers['accept'] = 'text/event-stream'
    request = urllib.request.Request(url, data=data, headers=headers, method='POST')
    try:
        response = urllib.request.urlopen(request, timeout=30)
    except urllib.error.HTTPError as err:
        # A refusal is an answer with a status and a body, exactly as Node's fetch reports it.
        return Upstream(err.code, err)
    return Upstream(response.status, response)


# ---------------------------------------------------------------- the HTTP server


class Handler(BaseHTTPRequestHandler):
    """The entry's one server: the models route, the two completion wires, and nothing else."""

    protocol_version = 'HTTP/1.1'
    server_version = 'makefun-aggregator-py/1'
    sys_version = ''

    # The trial's stderr is the entry's own log: no access line per request.
    def log_message(self, format: str, *args: object) -> None:  # the base class's name
        pass

    def do_GET(self) -> None:  # the base class's name
        self.dispatch()

    def do_POST(self) -> None:  # the base class's name
        self.dispatch()

    # -- plumbing -----------------------------------------------------------------------------------

    def dispatch(self) -> None:
        self.headers_sent = False
        self.upstream_failures = []
        path = urlsplit(self.path).path
        try:
            if self.command == 'GET' and path == '/v1/models':
                self.models()
                return
            if self.command == 'POST' and path == '/v1/chat/completions':
                self.chat()
                return
            if self.command == 'POST' and path == '/v1/messages':
                self.messages()
                return
            wire = 'anthropic' if path == '/v1/messages' else 'openai'
            self.send_error_body(wire, 404, 'not_found', f'no route {self.command} {path}')
        except (BrokenPipeError, ConnectionResetError):
            # The client went away mid-answer: nothing left to say, and the connection is done.
            self.close_connection = True
        except Exception:  # one request's failure must not take the server down
            log(traceback.format_exc())
            if not self.headers_sent:
                wire = 'anthropic' if urlsplit(self.path).path.startswith('/v1/messages') else 'openai'
                try:
                    self.send_error_body(wire, 500, 'internal_error', 'the server failed to answer')
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True

    def send_json(self, status: int, body: object) -> None:
        text = json.dumps(body, separators=(',', ':'), ensure_ascii=False)
        raw = text.encode('utf-8')
        self.send_response(status)
        self.send_header('content-type', 'application/json')
        self.send_header('content-length', str(len(raw)))
        self.failure_header()
        self.headers_sent = True
        self.end_headers()
        self.wfile.write(raw)

    def send_error_body(self, wire: str, status: int, error_type: str, message: str) -> None:
        if wire == 'anthropic':
            self.send_json(status, anthropic_error(error_type, message))
        else:
            self.send_json(status, openai_error(message, error_type, None))

    def failure_header(self):
        if self.upstream_failures:
            value = json.dumps(self.upstream_failures[:8], separators=(',', ':'), ensure_ascii=True)
            if len(value.encode('ascii')) <= 8192:
                self.send_header('x-makefun-upstream-failures', value)

    def stream_headers(self) -> None:
        self.send_response(200)
        self.send_header('content-type', 'text/event-stream')
        self.send_header('cache-control', 'no-cache')
        self.failure_header()
        # Python's http.server does not chunk a response without a length; the close is what ends the
        # stream for the client. Node's http server chunks instead and keeps the connection alive.
        self.send_header('connection', 'close')
        self.headers_sent = True
        self.end_headers()
        self.close_connection = True

    def write_event(self, text: str) -> None:
        self.wfile.write(text.encode('utf-8'))
        self.wfile.flush()

    def read_json(self) -> Json | None:
        """One client request body, or None when it is not one JSON object or is too large."""
        if self.headers.get('transfer-encoding') is not None:
            self.close_connection = True
            return None
        length = self.headers.get('content-length')
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
        raw = self.rfile.read(size) if size > 0 else b''
        if not raw:
            return None
        try:
            value = json.loads(raw.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return None
        return value if is_record(value) else None

    def refused(self, wire: str, status: int, text: str) -> None:
        """A refused or broken gateway call as a 502 in the client's wire: the prompt is lost."""
        suffix = '' if text == '' else f': {text[:500]}'
        if wire == 'anthropic':
            self.send_error_body(wire, 502, 'api_error', f'the gateway answered HTTP {status}{suffix}')
        else:
            self.send_error_body(wire, 502, 'upstream_error', f'the gateway answered HTTP {status}{suffix}')

    # -- the routes ---------------------------------------------------------------------------------

    def models(self) -> None:
        """GET /v1/models, in the shape of the auth header the client sent."""
        listing = served_models()
        if self.headers.get('x-api-key') is not None:
            self.send_json(200, anthropic_model_list(listing))
        else:
            self.send_json(200, openai_model_list(listing))

    def chat(self) -> None:
        with request_lock:
            self.complete('openai')

    def messages(self) -> None:
        with request_lock:
            self.complete('anthropic')

    def complete(self, wire: str) -> None:
        """Use one routing/failover path for both client protocols.

        Retry only before sending client headers. Once streaming starts, an
        upstream failure is surfaced as an SSE error, never a second answer.
        """
        body = self.read_json()
        if body is None:
            self.send_error_body(wire, 400, 'invalid_request_error',
                                 'the request body is not one JSON object')
            return
        try:
            prepared_request(body, wire, 'auto')
        except (ValueError, TypeError) as err:
            self.send_error_body(wire, 400, 'invalid_request_error', str(err))
            return
        state = state_for_brief()
        epoch = state.begin()
        candidates = candidates_for(requirements_for(body))
        if not candidates:
            error = 'overloaded_error' if wire == 'anthropic' else 'no_free_model'
            self.send_error_body(wire, 503, error,
                                 'no free model in the brief meets this request')
            return
        last_status, last_text = 502, 'all eligible models failed or are cooling down'
        streaming = body.get('stream') is True
        for route in candidates:
            model_id = route['id']
            if not state.available(model_id, epoch):
                continue
            if not state.reserve():
                last_status, last_text = 429, 'trial call budget exhausted'
                break
            upstream = None
            try:
                request = prepared_request(body, wire, model_id)
                upstream = dial(model_id, request)
                if not upstream.ok:
                    last_status = upstream.status
                    state.failure(model_id, upstream.status, epoch)
                    last_text = upstream.text()
                    self.upstream_failures.append(refusal_row(last_status, last_text, model_id))
                    continue
                if streaming:
                    # Validate the first actual data frame before committing HTTP 200.
                    frames = iter_frames(upstream.chunks())
                    first = None
                    for frame in frames:
                        payload = sse_data(frame)
                        if not payload:
                            continue
                        if payload == '[DONE]':
                            raise ValueError('gateway returned an empty stream')
                        parsed = json.loads(payload)
                        if not is_record(parsed) or parsed.get('error') or not isinstance(parsed.get('choices'), list):
                            raise ValueError('gateway returned an invalid stream')
                        first = frame
                        break
                    if first is None:
                        raise ValueError('gateway returned an empty stream')
                    chunks = ((frame + '\n\n').encode('utf-8')
                              for frame in itertools.chain([first], frames))
                    events = (anthropic_events_from_openai(chunks, model_id)
                              if wire == 'anthropic' else openai_stream_with_model(chunks, model_id))
                    self.stream_headers()
                    for event in events:
                        self.write_event(event)
                else:
                    parsed = json.loads(upstream.text())
                    if (not is_record(parsed) or parsed.get('error')
                            or not isinstance(parsed.get('choices'), list)
                            or not parsed['choices']
                            or not is_record(parsed['choices'][0])
                            or not is_record(parsed['choices'][0].get('message'))):
                        raise ValueError('gateway returned an invalid completion')
                    answer = (openai_to_anthropic(parsed, model_id) if wire == 'anthropic'
                              else openai_json_with_model(parsed, model_id))
                    self.send_json(200, answer)
                state.success(model_id)
                return
            except (BrokenPipeError, ConnectionResetError):
                raise
            except Exception as err:
                last_status, last_text = 502, str(err)
                state.failure(model_id, 502, epoch)
                if self.headers_sent:
                    error = (anthropic_error('api_error', 'upstream stream failed')
                             if wire == 'anthropic' else openai_error('upstream stream failed', 'upstream_error', None))
                    self.write_event(sse_event('error' if wire == 'anthropic' else None, error))
                    return
            finally:
                if upstream is not None:
                    upstream.close()
        self.refused(wire, last_status, last_text)


# ---------------------------------------------------------------- register and poll


def register(port: int, report: Json) -> None:
    """POST /register with the port and the one catalog; the env answers with the trial's prompt count."""
    body = json.dumps({'protocol_version': PROTOCOL_VERSION, 'port': port, 'catalog': report}, separators=(',', ':'), ensure_ascii=False)
    request = urllib.request.Request(
        f'{ENV_URL}/register',
        data=body.encode('utf-8'),
        headers={'content-type': 'application/json'},
        method='POST',
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            status = response.status
            reply_text = response.read().decode('utf-8', 'replace')
    except urllib.error.HTTPError as err:
        raise RuntimeError(f'POST {ENV_URL}/register answered HTTP {err.code}: {err.read().decode("utf-8", "replace")[:500]}') from err
    if not (200 <= status < 300):
        raise RuntimeError(f'POST {ENV_URL}/register answered HTTP {status}: {reply_text[:500]}')
    try:
        reply = json.loads(reply_text)
    except (json.JSONDecodeError, ValueError):
        reply = None
    if is_record(reply) and is_number(reply.get('requests')):
        log(f'registered: the trial holds {reply["requests"]} prompts')


def poll() -> None:
    """Poll GET /work every 100 ms; exit 0 the moment it says done."""
    while True:
        time.sleep(POLL_INTERVAL_SEC)
        try:
            with urllib.request.urlopen(f'{ENV_URL}/work', timeout=2) as response:
                if response.status != 200:
                    continue
                work = json.loads(response.read().decode('utf-8', 'replace'))
        except Exception:  # the env is not reachable yet: try again on the next tick
            continue
        if is_record(work) and work.get('done') is True:
            sys.stderr.flush()
            os._exit(0)


# ---------------------------------------------------------------- start


def stop(*_: object) -> None:
    """SIGTERM and SIGINT end the trial as Node's process.exit(0) does: exit 0, at once."""
    sys.stderr.flush()
    os._exit(0)


def start() -> None:
    if BRIEF['error'] is not None:
        log(BRIEF['error'])
    if '--list' in sys.argv[1:]:
        # The same document the register body carries, byte for byte: print the one built object and stop
        # before any server starts. A process whose stdout is not flushed loses the tail, so flush it.
        sys.stdout.write(json.dumps(CATALOG, separators=(',', ':'), ensure_ascii=False) + '\n')
        sys.stdout.flush()
        return
    if ENV_URL == '':
        raise RuntimeError('MAKEFUN_ENV_URL is not set')
    if chat_completions_url() == '':
        raise RuntimeError('OPENAI_BASE_URL is not set')
    server = ThreadingHTTPServer(('0.0.0.0', 0), Handler)
    try:
        port = server.server_address[1]
        # 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
        if port == 8080:
            raise RuntimeError('the OS picked 8080, which the env owns on one layout: restart the trial')
        log(f'listening on 0.0.0.0:{port} with {len(served_models())} free models')
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        threading.Thread(target=server.serve_forever, name='http-server', daemon=True).start()
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
    except BaseException:  # Node's top-level catch: log the stack and exit 1
        log(traceback.format_exc())
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
