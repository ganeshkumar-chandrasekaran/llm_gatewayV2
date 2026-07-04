# LLM Gateway V2

A multi-provider LLM gateway with **auto-failover**, **rate limiting**, **MCP agent tools**, **prompt caching**, **tool calling**, **structured output**, and a built-in **mock provider** for zero-config development.

**Works out of the box** — no API keys required. The mock provider lets you test the full agent + MCP tool pipeline immediately.

---

## Features

- **7 LLM providers** — Ollama, Gemini, NVIDIA NIM, Groq, Cerebras, OpenRouter, GitHub Models
- **Mock provider** — built-in simulated LLM for testing without API keys
- **MCP agent endpoint** — agentic loop that connects to an MCP tool server (calculator, notes, time, temperature)
- **Auto-failover** — if one provider is rate-limited or down, the gateway automatically tries the next
- **Rate limiting** — RPM/RPD/TPM tracking with cooldowns per provider
- **Capability-aware routing** — skips providers that lack required features (tools, caching, reasoning)
- **Prompt caching** — Gemini explicit cache + implicit prefix caching for OpenAI-compatible providers
- **Tool calling** — native tool-use translated per provider, canonical `tool_calls[]` response
- **Structured output** — JSON schema validation with corrective retry
- **Reasoning budgets** — `reasoning="off"|"low"|"medium"|"high"` mapped per provider
- **Web dashboard** — real-time monitoring, interactive testing, and MCP tool explorer

---

## Quick Start (No API Keys Needed)

```bash
# 1. Clone the repo
git clone https://github.com/ganeshkumar-chandrasekaran/llm_gatewayV2.git
cd llm_gatewayV2

# 2. Run it
./run.sh                  # creates .venv, installs deps, starts on port 8100

# 3. Open the dashboard
open http://localhost:8100/static/dashboard.html
```

That's it! The **mock provider** is enabled by default. In the dashboard:

1. Select **mock** from the provider dropdown
2. Make sure **Agent + MCP Tools** is checked
3. Click any sample prompt or type your own
4. Hit **Send** and see the tool trace

### Try These Prompts (Mock + MCP)

| Prompt | Tool Used | Result |
|--------|-----------|--------|
| `add 7 plus 5` | add(a=7, b=5) | 12.0 |
| `multiply 12 times 8` | multiply(a=12, b=8) | 96.0 |
| `divide 100 by 4` | divide(a=100, b=4) | 25.0 |
| `subtract 50 minus 18` | subtract(a=50, b=18) | 32.0 |
| `what is the square root of 144` | sqrt(number=144) | 12.0 |
| `3 raised to the power of 4` | power(base=3, exponent=4) | 81.0 |
| `what time is it now` | get_current_time() | current date/time |
| `convert 100 celsius to fahrenheit` | convert_temperature(100, C, F) | 212.0 |

### Using cURL

```bash
# Agent mode (with MCP tools)
curl -X POST http://localhost:8100/v1/agent \
  -H "Content-Type: application/json" \
  -d '{"prompt": "multiply 4 times 2", "provider": "mock"}'

# Chat mode (plain LLM, no tools)
curl -X POST http://localhost:8100/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Say hello", "provider": "mock"}'
```

---

## Connecting a Real LLM Provider

The mock provider is great for testing, but you'll want a real LLM for actual use. The gateway supports 7 providers — here's how to set up the most popular free ones.

### Step 1: Create the `.env` File

Copy the example file to the **parent directory** (one level above `llm_gatewayV2/`):

```bash
cp .env.example ../.env
```

### Step 2: Get Your API Key

#### Google Gemini (Recommended — generous free tier)

1. Go to [aistudio.google.com](https://aistudio.google.com/)
2. Click **Get API Key** → **Create API Key**
3. Copy the key (starts with `AIza...`)
4. Add to your `../.env`:

```env
GEMINI_API_KEY=AIzaSy...your_key_here
GEMINI_MODEL=gemini-2.5-flash
```

#### GitHub Models (Free with GitHub account)

1. Go to [github.com/marketplace/models](https://github.com/marketplace/models)
2. Pick any model → click **Use this model** to accept terms
3. Create a Personal Access Token at [github.com/settings/tokens](https://github.com/settings/tokens)
   - Click **Generate new token (classic)**
   - No special scopes needed — just the default
4. Add to your `../.env`:

```env
GITHUB_ACCESS_TOKEN=github_pat_...your_token_here
GITHUB_MODEL=openai/gpt-4.1-mini
```

#### Other Providers

| Provider | Get Key At | Env Variable | Default Model |
|----------|-----------|--------------|---------------|
| Groq | [console.groq.com/keys](https://console.groq.com/keys) | `GROQ_API_KEY` | `llama-3.3-70b-versatile` |
| NVIDIA NIM | [build.nvidia.com](https://build.nvidia.com) | `NVIDIA_API_KEY` | `deepseek-ai/deepseek-v3.2` |
| Cerebras | [cloud.cerebras.ai](https://cloud.cerebras.ai) | `CEREBRAS_API_KEY` | `qwen-3-235b-a22b-instruct-2507` |
| OpenRouter | [openrouter.ai/keys](https://openrouter.ai/keys) | `OPEN_ROUTER_API_KEY` | `nvidia/nemotron-3-super-120b-a12b:free` |
| Ollama | [ollama.ai](https://ollama.ai) (local) | `OLLAMA_MODEL` | (your installed model) |

### Step 3: Restart the Gateway

```bash
# Stop the running server (Ctrl+C), then:
./run.sh
```

The new provider will automatically appear in the dashboard. The failover order is controlled by `LLM_ORDER` in `.env`.

---

## How the Agent + MCP Flow Works

When you send a prompt to `/v1/agent`, here's what happens:

```
User: "multiply 4 times 2"
  │
  ▼
┌─────────────────────────────┐
│  /v1/agent endpoint         │
│  1. Spawns MCP server       │
│  2. Discovers 13 tools      │
│  3. Sends prompt + tools    │
│     to LLM provider         │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  LLM Provider (mock/gemini) │
│  Decides: call multiply     │
│  with a=4, b=2              │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  MCP Server (mcp_server.py) │
│  Executes multiply(4, 2)    │
│  Returns: {"result": 8.0}   │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  LLM formats final answer   │
│  "The result is 8.0."       │
└─────────────────────────────┘
```

### Available MCP Tools

| Tool | Description | Example |
|------|-------------|---------|
| `add` | Add two numbers | add(a=7, b=5) → 12 |
| `subtract` | Subtract b from a | subtract(a=10, b=3) → 7 |
| `multiply` | Multiply two numbers | multiply(a=4, b=2) → 8 |
| `divide` | Divide a by b | divide(a=100, b=4) → 25 |
| `power` | Raise base to exponent | power(base=2, exponent=10) → 1024 |
| `sqrt` | Square root | sqrt(number=144) → 12 |
| `save_note` | Save a note with title | save_note(title="todo", content="buy milk") |
| `read_note` | Read a note by title | read_note(title="todo") |
| `list_notes` | List all note titles | list_notes() |
| `delete_note` | Delete a note | delete_note(title="todo") |
| `get_current_time` | Current date/time | get_current_time() |
| `string_length` | Count chars and words | string_length(text="hello world") |
| `convert_temperature` | Convert C/F/K | convert_temperature(100, "C", "F") → 212 |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/agent` | POST | Agent endpoint with MCP tool loop |
| `/v1/agent/tools` | GET | List available MCP tools |
| `/v1/chat` | POST | Direct LLM chat (no tools) |
| `/v1/status` | GET | Provider status, rate limits, usage |
| `/v1/providers` | GET | List configured providers |
| `/v1/capabilities` | GET | Per-provider capability matrix |
| `/v1/calls` | GET | Recent call logs |
| `/static/dashboard.html` | GET | Web dashboard |

---

## Project Structure

```
llm_gatewayV2/
├── main.py              # FastAPI app, routes, agent loop
├── providers.py         # LLM provider adapters (Gemini, GitHub, Mock, etc.)
├── router.py            # Rate limiting + capability-aware failover
├── mcp_server.py        # MCP tool server (calculator, notes, utilities)
├── schemas.py           # Pydantic v2 request/response models
├── cache.py             # Gemini prompt caching (SHA-256 keyed)
├── db.py                # SQLite call logging
├── client.py            # Python SDK
├── run.sh               # Setup + start script
├── requirements.txt     # Python dependencies
├── .env.example         # Template for API keys
├── static/
│   ├── dashboard.html   # Web dashboard with testing UI
│   └── help.html        # Provider setup guide
└── tests/
    └── test_all_providers.py  # Per-provider test matrix
```

---

## Configuration

All configuration is via environment variables in `../.env` (parent directory):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_ORDER` | `mock,ollama,gemini,...` | Failover order (comma-separated) |
| `GATEWAY_V2_PORT` | `8100` | Server port |
| `GEMINI_API_KEY` | — | Google AI Studio key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model |
| `GITHUB_ACCESS_TOKEN` | — | GitHub PAT |
| `GITHUB_MODEL` | `openai/gpt-4.1-mini` | GitHub Models model |
| `GROQ_API_KEY` | — | Groq key |
| `NVIDIA_API_KEY` | — | NVIDIA NIM key |
| `CEREBRAS_API_KEY` | — | Cerebras key |
| `OPEN_ROUTER_API_KEY` | — | OpenRouter key |
| `OLLAMA_MODEL` | — | Ollama model name |

**No keys = mock provider only.** Add any key and the provider joins the failover chain automatically.

---

## License

MIT
