import os, time, json
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from jsonschema import Draft202012Validator, ValidationError

ROOT = Path(__file__).parent
load_dotenv(ROOT.parent / ".env")

import db
import providers as P
from router import Router, LIMITS, SHORTCUTS, resolve
from cache import GeminiCache
from schemas import ChatRequest, ChatResponse, ToolCall

DEFAULT_ORDER = ["mock", "ollama", "gemini", "nvidia", "groq", "cerebras", "openrouter", "github"]
ORDER = [x.strip() for x in os.getenv("LLM_ORDER", ",".join(DEFAULT_ORDER)).split(",") if x.strip()]
PORT = int(os.getenv("GATEWAY_V2_PORT", "8100"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init()
    app.state.cache = GeminiCache(ttl_seconds=300)
    app.state.providers = P.build_providers(app.state.cache)
    app.state.router = Router(app.state.providers, ORDER)
    yield


app = FastAPI(title="LLM Gateway V2", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


def _normalize_messages(req: ChatRequest):
    if req.messages:
        return list(req.messages)
    msgs = []
    msgs.append({"role": "user", "content": req.prompt or ""})
    return msgs


def _system_blocks(req: ChatRequest):
    """Returns the system_blocks payload to hand to the provider adapter."""
    if req.system is None:
        return None
    if isinstance(req.system, str):
        if req.cache_system:
            return [{"text": req.system, "cache": True}]
        return req.system
    return [b.model_dump() if hasattr(b, "model_dump") else b for b in req.system]


def _est_tokens(messages, system_blocks, max_tokens):
    chars = sum(len(str(m.get("content", ""))) for m in messages)
    if isinstance(system_blocks, str):
        chars += len(system_blocks)
    elif isinstance(system_blocks, list):
        for b in system_blocks:
            chars += len(b.get("text", "") if isinstance(b, dict) else "")
    return chars // 4 + max_tokens


def _backoff_for(err: Exception, has_model_override: bool = False):
    msg = str(err).lower()
    status = getattr(err, "status", None)
    if status == 429:
        if "queue" in msg: return 15, "server queue full"
        if "quota" in msg or "rpm" in msg or "per minute" in msg: return 60, "RPM quota burned"
        if "rpd" in msg or "per day" in msg or "daily" in msg: return 3600, "RPD quota burned"
        return 30, "rate limited"
    if status and 500 <= status < 600: return 20, f"upstream {status}"
    if status == 408 or "timeout" in msg: return 10, "timeout"
    if status in (401, 403):
        # When the caller explicitly picked a model, 403/404 likely means
        # "this model not available to your account" rather than "key dead".
        # Don't blackball the whole provider for 10 minutes.
        if has_model_override:
            return 0, ""
        return 600, "auth error"
    if status == 404 and has_model_override:
        return 0, ""
    return 0, ""


def _attempts_str(attempts):
    return "; ".join(f"{a['provider']}:{a['reason']}" for a in attempts)


def _required_caps(req: ChatRequest):
    caps = []
    if req.tools: caps.append("tools")
    if req.reasoning and req.reasoning != "off": caps.append("reasoning")
    if req.response_format: caps.append("structured")
    return caps


def _validate_structured(text: str, schema: dict):
    try:
        obj = json.loads(text)
    except Exception as e:
        raise ValueError(f"output is not JSON: {e}")
    Draft202012Validator(schema).validate(obj)
    return obj


@app.post("/v1/chat")
async def chat(req: ChatRequest):
    router = app.state.router
    messages = _normalize_messages(req)
    system_blocks = _system_blocks(req)
    prompt_text = "".join(str(m.get("content", "")) for m in messages)
    est = _est_tokens(messages, system_blocks, req.max_tokens)
    explicit_override = bool(req.provider)
    required_caps = _required_caps(req)

    candidates = router.candidates(req.provider) if req.provider else list(router.order)
    if req.provider and not candidates:
        raise HTTPException(400, f"unknown provider '{req.provider}'. Try one of: {list(router.providers)} or shortcuts {list(SHORTCUTS)}")

    all_attempts = []
    last_err = None

    # When explicit provider is requested and the only blocker is cooldown,
    # wait briefly rather than 503-ing — this is what users intuitively expect.
    if explicit_override and len(candidates) == 1:
        import asyncio as _asyncio
        deadline = time.time() + 30
        while time.time() < deadline:
            name, _ = router.pick(est, candidates, required_caps=required_caps)
            if name is not None:
                break
            cd = router.state[candidates[0]].snapshot(LIMITS[candidates[0]])["cooldown_remaining"]
            if cd <= 0 or cd > 30:
                break
            await _asyncio.sleep(min(cd + 0.05, 5))

    for _ in range(len(candidates) + 1):
        name, atts = router.pick(est, candidates, required_caps=required_caps)
        all_attempts.extend(atts)
        if name is None:
            break

        provider = router.providers[name]
        t0 = time.time()
        router.state[name].record(0)

        try:
            if req.stream:
                async def gen():
                    try:
                        agg = []
                        async for chunk in provider.stream(messages,
                                                          max_tokens=req.max_tokens,
                                                          temperature=req.temperature,
                                                          model=req.model,
                                                          tools=req.tools,
                                                          tool_choice=req.tool_choice,
                                                          reasoning=req.reasoning,
                                                          response_format=req.response_format,
                                                          system_blocks=system_blocks,
                                                          cache_system=bool(req.cache_system)):
                            agg.append(chunk)
                            if chunk.startswith("[[TOOL_CALL_DELTA]]"):
                                yield f"data: {json.dumps({'provider': name, 'tool_call_delta': chunk[len('[[TOOL_CALL_DELTA]] '):]})}\n\n"
                            else:
                                yield f"data: {json.dumps({'provider': name, 'delta': chunk})}\n\n"
                        text = "".join(agg)
                        latency = int((time.time() - t0) * 1000)
                        db.log_call(provider=name, model=req.model or provider.model,
                                    latency_ms=latency, status="ok",
                                    prompt_chars=len(prompt_text), response_chars=len(text),
                                    override=req.provider, attempted=_attempts_str(all_attempts))
                        yield f"data: {json.dumps({'done': True, 'provider': name})}\n\n"
                    except Exception as e:
                        db.log_call(provider=name, model=req.model or provider.model,
                                    status="error", error=str(e)[:500],
                                    latency_ms=int((time.time() - t0) * 1000),
                                    prompt_chars=len(prompt_text),
                                    override=req.provider, attempted=_attempts_str(all_attempts))
                        yield f"data: {json.dumps({'error': str(e)[:300]})}\n\n"
                return StreamingResponse(gen(), media_type="text/event-stream")

            result = await provider.chat(messages,
                                         max_tokens=req.max_tokens,
                                         temperature=req.temperature,
                                         model=req.model,
                                         tools=req.tools,
                                         tool_choice=req.tool_choice,
                                         reasoning=req.reasoning,
                                         response_format=req.response_format,
                                         system_blocks=system_blocks,
                                         cache_system=bool(req.cache_system))
            latency = int((time.time() - t0) * 1000)

            # Optional: validate structured output and (single) retry on failure.
            parsed = None
            if req.response_format and req.response_format.schema_ and not result["tool_calls"]:
                try:
                    parsed = _validate_structured(result["text"], req.response_format.schema_)
                except (ValueError, ValidationError) as ve:
                    # one corrective retry
                    fix_msgs = list(messages) + [
                        {"role": "assistant", "content": result["text"]},
                        {"role": "user", "content": f"Your previous reply did not match the required JSON schema: {ve}. Reply ONLY with valid JSON conforming to the schema."},
                    ]
                    result = await provider.chat(fix_msgs,
                                                 max_tokens=req.max_tokens,
                                                 temperature=0,
                                                 model=req.model,
                                                 response_format=req.response_format,
                                                 system_blocks=system_blocks,
                                                 cache_system=bool(req.cache_system))
                    try:
                        parsed = _validate_structured(result["text"], req.response_format.schema_)
                    except (ValueError, ValidationError) as ve2:
                        raise HTTPException(503, f"structured output failed validation: {ve2}")

            tokens = (result["input_tokens"] or 0) + (result["output_tokens"] or 0)
            router.state[name].tokens_today += tokens
            router.state[name].tokens_minute.append((time.time(), tokens))
            db.log_call(provider=name, model=result["model"],
                        input_tokens=result["input_tokens"], output_tokens=result["output_tokens"],
                        cache_create_tokens=result["cache_creation_input_tokens"],
                        cache_read_tokens=result["cache_read_input_tokens"],
                        latency_ms=latency, status="ok",
                        prompt_chars=len(prompt_text), response_chars=len(result["text"]),
                        override=req.provider, attempted=_attempts_str(all_attempts),
                        tool_calls=len(result["tool_calls"]),
                        reasoning_applied=result["reasoning_applied"],
                        tool_dialect=result["tool_call_dialect"])
            return ChatResponse(
                provider=name,
                model=result["model"],
                text=result["text"],
                tool_calls=[ToolCall(**tc) for tc in result["tool_calls"]],
                stop_reason=result["stop_reason"],
                input_tokens=result["input_tokens"],
                output_tokens=result["output_tokens"],
                cache_creation_input_tokens=result["cache_creation_input_tokens"],
                cache_read_input_tokens=result["cache_read_input_tokens"],
                latency_ms=latency,
                tool_call_dialect=result["tool_call_dialect"],
                reasoning_applied=result["reasoning_applied"],
                parsed=parsed,
                attempted=all_attempts,
            ).model_dump()

        except P.ProviderError as e:
            last_err = str(e)
            secs, reason = _backoff_for(e, has_model_override=bool(req.model))
            if secs > 0:
                router.state[name].mark_unavailable(secs, reason)
            db.log_call(provider=name, model=req.model or provider.model,
                        status="error", error=str(e)[:500],
                        latency_ms=int((time.time() - t0) * 1000),
                        prompt_chars=len(prompt_text),
                        override=req.provider, attempted=_attempts_str(all_attempts))
            tag = f"failed: {str(e)[:100]}"
            if secs > 0: tag += f" → backoff {secs:.0f}s ({reason})"
            all_attempts.append({"provider": name, "reason": tag})
            if explicit_override or not getattr(e, "retryable", True):
                raise HTTPException(502, f"{name} failed: {e}")
            candidates = [c for c in candidates if c != name]
            continue
        except HTTPException:
            raise
        except Exception as e:
            last_err = str(e)
            secs, reason = _backoff_for(e, has_model_override=bool(req.model))
            if secs > 0:
                router.state[name].mark_unavailable(secs, reason)
            db.log_call(provider=name, model=req.model or provider.model,
                        status="error", error=str(e)[:500],
                        latency_ms=int((time.time() - t0) * 1000),
                        prompt_chars=len(prompt_text),
                        override=req.provider, attempted=_attempts_str(all_attempts))
            all_attempts.append({"provider": name, "reason": f"exception: {str(e)[:120]}"})
            if explicit_override:
                raise HTTPException(502, f"{name} failed: {e}")
            candidates = [c for c in candidates if c != name]
            continue

    raise HTTPException(503, f"all providers unavailable. attempts: {all_attempts}. last_error: {last_err}")


@app.get("/v1/providers")
async def list_providers():
    r = app.state.router
    return {
        "order": r.order,
        "providers": list(r.providers.keys()),
        "shortcuts": SHORTCUTS,
        "limits": LIMITS,
        "models": {n: p.model for n, p in r.providers.items()},
    }


@app.get("/v1/capabilities")
async def capabilities():
    r = app.state.router
    out = {}
    for name, p in r.providers.items():
        caps = dict(getattr(p, "capabilities", {}))
        # per-model overrides
        caps = P.model_capabilities(name, p.model, caps)
        caps["model"] = p.model
        caps.update({
            "max_ctx": LIMITS[name]["max_ctx"],
            "rpm": LIMITS[name]["rpm"],
            "rpd": LIMITS[name]["rpd"],
        })
        out[name] = caps
    return out


@app.get("/v1/status")
async def status():
    r = app.state.router
    return {"order": r.order, "live": r.all_status(), "today": db.aggregate(), "limits": LIMITS}


@app.get("/v1/calls")
async def calls(limit: int = 100, provider: Optional[str] = None, status: Optional[str] = None):
    return db.recent(limit=limit, provider=provider, status=status)


# ────────────────────────────────────────────────────────────────────────────
# Agentic endpoint — LLM + MCP tool loop
# ────────────────────────────────────────────────────────────────────────────

import sys
from pydantic import BaseModel as _BM, Field as _F


class AgentRequest(_BM):
    prompt: str
    provider: Optional[str] = None
    model: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.7
    system: Optional[str] = None
    max_iterations: int = 5


class AgentResponse(_BM):
    final_text: str
    provider: str = ""
    model: str = ""
    iterations: int = 0
    tool_trace: list[dict] = _F(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_ms: int = 0


MCP_SERVER_PATH = str(ROOT / "mcp_server.py")
MCP_PYTHON = str(ROOT / ".venv" / "bin" / "python")


async def _call_llm(messages, system_text, req, mcp_tools):
    """Call the LLM via the gateway's internal provider, bypassing HTTP."""
    import asyncio as _aio
    router = app.state.router
    candidates = router.candidates(req.provider) if req.provider else list(router.order)
    est = _est_tokens(messages, system_text, req.max_tokens)

    if req.provider and len(candidates) == 1:
        deadline = time.time() + 30
        while time.time() < deadline:
            name, _ = router.pick(est, candidates, required_caps=["tools"])
            if name is not None:
                break
            cd = router.state[candidates[0]].snapshot(LIMITS[candidates[0]])["cooldown_remaining"]
            if cd <= 0 or cd > 30:
                break
            await _aio.sleep(min(cd + 0.1, 5))

    name, atts = router.pick(est, candidates, required_caps=["tools"])
    if name is None:
        raise Exception(f"No provider available: {atts}")

    provider = router.providers[name]
    router.state[name].record(0)
    t0 = time.time()
    result = await provider.chat(
        messages,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        model=req.model,
        tools=mcp_tools,
        tool_choice="auto",
        system_blocks=system_text,
    )
    latency = int((time.time() - t0) * 1000)
    tokens = (result["input_tokens"] or 0) + (result["output_tokens"] or 0)
    router.state[name].tokens_today += tokens
    router.state[name].tokens_minute.append((time.time(), tokens))
    db.log_call(
        provider=name, model=result["model"],
        input_tokens=result["input_tokens"], output_tokens=result["output_tokens"],
        latency_ms=latency, status="ok",
        override=req.provider, tool_calls=len(result["tool_calls"]),
    )
    result["provider"] = name
    result["latency_ms"] = latency
    return result


async def _run_agent_loop(req: AgentRequest) -> AgentResponse:
    """Connect to the MCP server, discover tools, and run the LLM-tool loop."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    import asyncio as _aio

    server_params = StdioServerParameters(
        command=MCP_PYTHON,
        args=[MCP_SERVER_PATH],
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            mcp_tools = []
            for t in tools_result.tools:
                mcp_tools.append({
                    "name": t.name,
                    "description": t.description or "",
                    "input_schema": t.inputSchema if hasattr(t, "inputSchema") else {},
                })

            system_text = req.system or "You are a helpful assistant with access to tools. Use tools when the user's request requires computation, data lookup, or actions. Always use the appropriate tool rather than guessing answers."
            messages = [{"role": "user", "content": req.prompt}]
            tool_trace = []
            total_in = 0
            total_out = 0
            total_latency = 0
            last_provider = ""
            last_model = ""
            resp_text = ""

            for iteration in range(req.max_iterations):
                try:
                    result = await _call_llm(messages, system_text, req, mcp_tools)
                except Exception as e:
                    return AgentResponse(
                        final_text=f"LLM call failed at iteration {iteration + 1}: {e}",
                        provider=last_provider, model=last_model,
                        iterations=iteration + 1, tool_trace=tool_trace,
                        total_input_tokens=total_in, total_output_tokens=total_out,
                        total_latency_ms=total_latency,
                    )

                last_provider = result.get("provider", "")
                last_model = result.get("model", "")
                total_in += result.get("input_tokens", 0)
                total_out += result.get("output_tokens", 0)
                total_latency += result.get("latency_ms", 0)

                tool_calls = result.get("tool_calls", [])
                resp_text = result.get("text", "")

                if not tool_calls:
                    return AgentResponse(
                        final_text=resp_text,
                        provider=last_provider, model=last_model,
                        iterations=iteration + 1, tool_trace=tool_trace,
                        total_input_tokens=total_in, total_output_tokens=total_out,
                        total_latency_ms=total_latency,
                    )

                assistant_msg = {"role": "assistant", "content": resp_text or "", "tool_calls": tool_calls}
                messages.append(assistant_msg)

                for tc in tool_calls:
                    tool_name = tc.get("name", "")
                    tool_args = tc.get("arguments", {})
                    tool_id = tc.get("id", "")

                    try:
                        mcp_result = await session.call_tool(tool_name, tool_args)
                        tool_output = mcp_result.content[0].text if mcp_result.content else "{}"
                    except Exception as e:
                        tool_output = json.dumps({"error": str(e)})

                    tool_trace.append({
                        "iteration": iteration + 1,
                        "tool": tool_name,
                        "arguments": tool_args,
                        "result": tool_output,
                    })

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "tool_name": tool_name,
                        "content": tool_output,
                    })

            return AgentResponse(
                final_text=resp_text or "Max iterations reached.",
                provider=last_provider, model=last_model,
                iterations=req.max_iterations, tool_trace=tool_trace,
                total_input_tokens=total_in, total_output_tokens=total_out,
                total_latency_ms=total_latency,
            )


@app.post("/v1/agent")
async def agent_endpoint(req: AgentRequest):
    try:
        result = await _run_agent_loop(req)
        return result.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Agent error: {e}")


@app.get("/v1/agent/tools")
async def list_mcp_tools():
    """List all tools available from the MCP server."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_params = StdioServerParameters(command=MCP_PYTHON, args=[MCP_SERVER_PATH])
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            return [{
                "name": t.name,
                "description": t.description or "",
                "input_schema": t.inputSchema if hasattr(t, "inputSchema") else {},
            } for t in tools_result.tools]


# ────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(str(ROOT / "static" / "dashboard.html"))


@app.get("/help", response_class=HTMLResponse)
async def help_page():
    return FileResponse(str(ROOT / "static" / "help.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
