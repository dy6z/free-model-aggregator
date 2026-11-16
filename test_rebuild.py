"""Rebuild regression tests. All gateway completions are synthetic test fixtures."""
import json
import os
import subprocess
import sys
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import main
import test_integration
from test_integration import CALLS, model

ROOT = Path(__file__).resolve().parent
KIT = ROOT.parent / 'makefun-verification/free-model-aggregator-02-kit'


class RoutingRegression(unittest.TestCase):
    setUpClass = classmethod(test_integration.Integration.setUpClass.__func__)
    tearDownClass = classmethod(test_integration.Integration.tearDownClass.__func__)
    setUp = test_integration.Integration.setUp
    request = test_integration.Integration.request
    body = test_integration.Integration.body

    def test_refusal_trail_and_cooldown(self):
        req = urllib.request.Request(self.base + '/v1/chat/completions',
                                     data=json.dumps(self.body()).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req) as response:
            trail = json.loads(response.headers['x-makefun-upstream-failures'])
            response.read()
        self.assertEqual([(r['status'], r['model']) for r in trail],
                         [(429, 'rate:free'), (503, 'down:free')])
        CALLS.clear()
        self.assertEqual(self.request('/v1/messages', self.body())[0], 200)
        self.assertEqual([r['model'] for r in CALLS], ['ok:free'])

    def test_quality_and_capability_union(self):
        main.BRIEF['models'] = [model('low:free', 1), model('high:free', 99, tools=False)]
        main.BRIEF['prompts'] = [{'requires': {}}]
        self.request('/v1/chat/completions', self.body(tools=[{'type': 'function', 'function': {'name': 'weather'}}]))
        self.assertEqual(CALLS[-1]['model'], 'low:free')
        self.request('/v1/chat/completions', self.body())
        self.assertEqual(CALLS[-1]['model'], 'high:free')

    def test_invalid_caps_no_gateway_call(self):
        for cap in (0, -1, 8193, True, 1.5):
            self.assertEqual(self.request('/v1/chat/completions', self.body(max_tokens=cap))[0], 400)
        self.assertEqual(CALLS, [])

    def test_call_budget(self):
        main.state_for_brief().calls = 60
        self.assertEqual(self.request('/v1/chat/completions', self.body())[0], 502)
        self.assertEqual(CALLS, [])

    def test_routing_independent_of_elapsed_time(self):
        traces = []
        for clock in (1.0, 1000000.0):
            main.routing_state = main.RoutingState(main.BRIEF)
            main.manifest_index = 0
            CALLS.clear()
            with patch('time.monotonic', return_value=clock):
                for _ in range(4):
                    self.assertEqual(self.request('/v1/chat/completions', self.body())[0], 200)
            traces.append([x['model'] for x in CALLS])
        self.assertEqual(traces[0], traces[1])
        self.assertEqual(traces[0], ['rate:free', 'down:free', 'ok:free',
                                    'ok:free', 'down:free', 'ok:free', 'rate:free', 'ok:free'])

    def test_stream_usage_requested(self):
        for path in ('/v1/chat/completions', '/v1/messages'):
            status, raw = self.request(path, self.body(stream=True))
            self.assertEqual(status, 200)
            self.assertTrue(CALLS[-1]['stream_options']['include_usage'])
            if path.endswith('/messages'):
                events = [json.loads(main.sse_data(f)) for f in raw.strip().split('\n\n')]
                delta = next(e for e in events if e['type'] == 'message_delta')
                self.assertEqual(delta['usage'], {'input_tokens': 8, 'output_tokens': 4})


class WireRegression(unittest.TestCase):
    def test_tool_history_and_choice(self):
        body = {'system': [{'type': 'text', 'text': 'system'}], 'max_tokens': 42,
                'tool_choice': {'type': 'none', 'disable_parallel_tool_use': True},
                'messages': [{'role': 'assistant', 'content': [
                    {'type': 'tool_use', 'id': 'c1', 'name': 'weather', 'input': {'city': 'Paris'}}]},
                    {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'c1', 'content': 'sunny'}]}]}
        out = main.prepared_request(body, 'anthropic', 'chosen')
        self.assertEqual(out['messages'][0], {'role': 'system', 'content': 'system'})
        self.assertEqual(out['messages'][-1], {'role': 'tool', 'tool_call_id': 'c1', 'content': 'sunny'})
        self.assertEqual(json.loads(out['messages'][1]['tool_calls'][0]['function']['arguments']), {'city': 'Paris'})
        self.assertEqual(out['tool_choice'], 'none')
        self.assertFalse(out['parallel_tool_calls'])

    def test_fragmented_unicode_and_tool_stream(self):
        chunks = [
            {'choices': [{'delta': {'content': 'é'}, 'finish_reason': None}]},
            {'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'c1', 'function': {'name': 'weather', 'arguments': '{"city":'}}]}, 'finish_reason': None}]},
            {'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': '"Paris"}'}}]}, 'finish_reason': 'tool_calls'}]},
            {'choices': [], 'usage': {'prompt_tokens': 7, 'completion_tokens': 9}}]
        raw = ''.join('data: ' + json.dumps(c, ensure_ascii=False) + '\r\n\r\n' for c in chunks).encode() + b'data: [DONE]\r\n\r\n'
        events = [json.loads(main.sse_data(e.strip())) for e in main.anthropic_events_from_openai((bytes([b]) for b in raw), 'chosen')]
        self.assertEqual(events[0]['type'], 'message_start')
        self.assertEqual(events[-1]['type'], 'message_stop')
        starts = [e for e in events if e['type'] == 'content_block_start']
        self.assertEqual(starts[1]['content_block']['id'], 'c1')
        fragments = [e['delta']['partial_json'] for e in events if e.get('delta', {}).get('type') == 'input_json_delta']
        self.assertEqual(json.loads(''.join(fragments)), {'city': 'Paris'})
        self.assertEqual(events[-2]['delta']['stop_reason'], 'tool_use')
        self.assertEqual(events[-2]['usage']['input_tokens'], 7)

    def test_truncated_stream_is_not_success(self):
        raw = b'data: {"choices":[{"delta":{"content":"partial"},"finish_reason":null}]}\n\n'
        for converter in (main.openai_stream_with_model, main.anthropic_events_from_openai):
            with self.assertRaises(ValueError):
                list(converter([raw], 'chosen'))


class Startup(unittest.TestCase):
    def test_register_list_and_done_exit(self):
        captured = {}
        class Env(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                captured['registration'] = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                base = 'http://127.0.0.1:' + str(captured['registration']['port'])
                for key in ('Authorization', 'x-api-key'):
                    with urllib.request.urlopen(urllib.request.Request(base + '/v1/models', headers={key: 'fixture'})) as response:
                        captured[key] = json.load(response)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"ok":true,"requests":0}')
            def do_GET(self):
                captured['poll'] = self.path
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"done":true}')
        server = ThreadingHTTPServer(('127.0.0.1', 0), Env)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        env = {k: v for k, v in os.environ.items() if not k.startswith(('MAKEFUN_', 'OPENAI_', 'ANTHROPIC_'))}
        env.update(MAKEFUN_PROMPT_FILE=str(KIT / 'practice/trial-0/input/prompt.txt'),
                   MAKEFUN_ENV_URL=f'http://127.0.0.1:{server.server_port}',
                   OPENAI_BASE_URL=f'http://127.0.0.1:{server.server_port}/v1', OPENAI_API_KEY='fixture')
        try:
            listing = subprocess.run([sys.executable, '-P', '-s', str(ROOT / 'main.py'), '--list'], env=env, capture_output=True, text=True, timeout=10)
            run = subprocess.run([sys.executable, '-P', '-s', str(ROOT / 'main.py')], env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(listing.returncode, 0, listing.stderr)
            registration = captured['registration']
            self.assertEqual(registration['protocol_version'], 1)
            self.assertNotEqual(registration['port'], 8080)
            self.assertEqual(json.dumps(registration['catalog'], separators=(',', ':'), ensure_ascii=False) + '\n', listing.stdout)
            ids = [m['id'] for m in registration['catalog']['providers'][0]['models']]
            self.assertEqual(len(set(ids)), 14)
            for key in ('Authorization', 'x-api-key'):
                self.assertEqual([m['id'] for m in captured[key]['data']], ids)
            self.assertEqual(captured['poll'], '/work')
        finally:
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    unittest.main()
