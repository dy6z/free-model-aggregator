# dy6z/free-model-aggregator

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/dy6z/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/dy6z/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

A self-contained Python HTTP server that aggregates free OpenRouter models for the make.fun `free-model-aggregator-02` bounty. It registers with the make.fun env, serves both OpenAI Chat Completions and Anthropic Messages wires, routes each prompt to the best-rated eligible free model, retries and fails over on 429/5xx, enforces per-model cooldowns, and reports upstream refusal trails in `x-makefun-upstream-failures`.

## Run it

```sh
python3 -P -s main.py
```

Only the Python standard library is required. Run with `--list` to print the registration catalog without starting the server.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
