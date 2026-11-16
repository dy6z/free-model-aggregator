"""Local HTTP integration tests; gateway answers are explicitly synthetic fixtures."""
import json
import threading
import unittest
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import main

CALLS = []

class Gateway(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        CALLS.append(body)
        model = body['model']
        status = 429 if model == 'rate:free' else 503 if model == 'down:free' else 200
        message = {'role': 'assistant', 'content': 'fixture answer'}
        if body.get('tools'):
            message['tool_calls'] = [{'id': 'call_1', 'type': 'function', 'function': {'name': 'weather', 'arguments': '{"city":"Paris"}'}}]
        finish = 'tool_calls' if body.get('tools') else 'stop'
        response = {'id': 'fixture', 'model': model, 'choices': [{'index': 0, 'message': message, 'finish_reason': finish}], 'usage': {'prompt_tokens': 8, 'completion_tokens': 4, 'total_tokens': 12}}
        if status != 200:
            raw = b'{"error":{"message":"injected failure"}}'
        elif body.get('stream'):
            chunks = [{'id': 'fixture', 'model': model, 'choices': [{'index': 0, 'delta': message, 'finish_reason': None}]}, {'id': 'fixture', 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}], 'usage': response['usage']}]
            raw = ''.join('data: '+json.dumps(c)+'\n\n' for c in chunks).encode() + b'data: [DONE]\n\n'
        else:
            raw = json.dumps(response).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'text/event-stream' if body.get('stream') else 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def model(name, rating, tools=True):
    return {'id': name, 'rating_milli': rating, 'free': True, 'tools': tools, 'vision': False, 'json': True, 'context_length': 32768, 'provider': 'openrouter', 'display_name': name, 'context': 32768}

class Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.servers = [ThreadingHTTPServer(('127.0.0.1', 0), Gateway), ThreadingHTTPServer(('127.0.0.1', 0), main.Handler)]
        main.GATEWAY_ROOT = 'http://127.0.0.1:'+str(cls.servers[0].server_port)
        main.GATEWAY_KEY = 'local-fixture'
        cls.base = 'http://127.0.0.1:'+str(cls.servers[1].server_port)
        for server in cls.servers:
            threading.Thread(target=server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        for server in cls.servers:
            server.shutdown()
            server.server_close()

    def setUp(self):
        CALLS.clear()
        main.BRIEF = {'models': [model('rate:free', 1000), model('down:free', 900), model('ok:free', 800)], 'prompts': [], 'fault': None}
        main.arrival_index = 0
        main.CATALOG = main.catalog_document(main.BRIEF)

    def request(self, path, body=None, headers=None):
        req = urllib.request.Request(self.base+path, data=json.dumps(body).encode() if body is not None else None, headers=headers or {'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()

    def body(self, **kw):
        return {'model': 'auto', 'messages': [{'role': 'user', 'content': 'hello'}], 'max_tokens': 32, **kw}

    def test_openai_failover(self):
        status, raw = self.request('/v1/chat/completions', self.body())
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)['model'], 'ok:free')
        self.assertEqual([x['model'] for x in CALLS], ['rate:free', 'down:free', 'ok:free'])

    def test_anthropic_failover_system(self):
        status, raw = self.request('/v1/messages', self.body(system='Keep concise'))
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)['content'][0]['text'], 'fixture answer')
        self.assertEqual(CALLS[-1]['messages'][0], {'role': 'system', 'content': 'Keep concise'})

    def test_streams(self):
        for path, marker in [('/v1/chat/completions', '[DONE]'), ('/v1/messages', 'message_stop')]:
            with self.subTest(path=path):
                status, raw = self.request(path, self.body(stream=True))
                self.assertEqual(status, 200)
                self.assertIn(marker, raw)
                self.assertIn('fixture answer', raw)
                self.assertIn('ok:free', raw)

    def test_anthropic_tools(self):
        status, raw = self.request('/v1/messages', self.body(tools=[{'name': 'weather', 'description': 'Weather', 'input_schema': {'type': 'object'}}]))
        self.assertEqual(status, 200)
        answer = json.loads(raw)
        self.assertEqual(answer['stop_reason'], 'tool_use')
        tool = next(x for x in answer['content'] if x['type'] == 'tool_use')
        self.assertEqual(tool['input'], {'city': 'Paris'})
        self.assertEqual(CALLS[-1]['tools'][0]['function']['name'], 'weather')

    def test_openai_system_json_tools(self):
        b = self.body(messages=[{'role': 'system', 'content': 'json only'}, {'role': 'user', 'content': 'hi'}], response_format={'type': 'json_object'}, tools=[{'type': 'function', 'function': {'name': 'weather', 'parameters': {'type': 'object'}}}])
        status, raw = self.request('/v1/chat/completions', b)
        self.assertEqual(status, 200)
        self.assertEqual(CALLS[-1]['response_format'], b['response_format'])
        self.assertEqual(CALLS[-1]['messages'], b['messages'])
        self.assertEqual(json.loads(raw)['choices'][0]['message']['tool_calls'][0]['id'], 'call_1')

    def test_catalog_both_wires(self):
        for headers in [{'Authorization': 'Bearer fixture'}, {'x-api-key': 'fixture'}]:
            status, raw = self.request('/v1/models', headers=headers)
            self.assertEqual(status, 200)
            self.assertEqual([m['id'] for m in json.loads(raw)['data']], ['rate:free', 'down:free', 'ok:free'])

    def test_no_capability(self):
        for m in main.BRIEF['models']:
            m['tools'] = False
        status, raw = self.request('/v1/messages', self.body(tools=[{'name': 'weather', 'input_schema': {}}]))
        self.assertEqual(status, 503)
        self.assertEqual(CALLS, [])

    def test_all_failed(self):
        main.BRIEF['models'] = main.BRIEF['models'][:2]
        status, raw = self.request('/v1/chat/completions', self.body())
        self.assertEqual(status, 502)
        self.assertIn('error', json.loads(raw))

if __name__ == '__main__':
    unittest.main()
