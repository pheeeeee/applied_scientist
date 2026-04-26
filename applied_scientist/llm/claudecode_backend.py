"""
ClaudeCodeBackend — subscription-based LLM backend for Applied Scientist.

Routes LLM calls through the `claude` CLI (Claude Code) instead of the Anthropic
API. This uses a Claude Pro/Max *subscription* rather than per-token API credits.

REQUIREMENTS
------------
- `claude` CLI installed and authenticated (https://docs.anthropic.com/en/docs/claude-code)
- Active Claude Pro/Max subscription linked to the CLI's OAuth credentials
- `CLAUDE_CONFIG_DIR` env var pointing to credentials directory (default: ~/.claude)

USAGE
-----
Set `backend: claudecode` in your config:

    llm:
      explorer:
        backend: claudecode
        model: haiku
      critic:
        backend: claudecode
        model: sonnet

LIMITATIONS
-----------
1. **Rate limits.** Consumer subscriptions are rate-limited for interactive human
   use. Running four agents continuously will hit those limits. Expect stalls.

2. **ToS risk.** Using a consumer subscription to drive an autonomous multi-agent
   system may violate Anthropic's acceptable use policy. Read the current terms.

3. **Subprocess overhead.** Each completion spawns a new `claude` process
   (~100-500 ms startup cost). Throughput is lower than direct API.

4. **Tool-call emulation.** We emulate tool calls by injecting tool schemas into
   the prompt and parsing JSON responses. Less faithful than native API tool use.

5. **Usage accounting.** Token counts may be zeros — Claude Code's JSON output
   doesn't always surface them. Subscription isn't billed per token anyway.

6. **No conversation state.** `--print` mode is stateless; multi-turn history is
   flattened into a single prompt.

INTENDED USE
------------
Development iteration, debugging, low-volume runs where API cost avoidance matters
more than throughput or reliability. NOT a drop-in replacement for AnthropicBackend
in production sweeps.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Optional

from applied_scientist.llm.base import LLMBackend, LLMMessage, LLMResponse


# Default location of Claude Code credentials.
DEFAULT_CLAUDE_CONFIG_DIR = os.path.expanduser("~/.claude")


class ClaudeCodeBackend(LLMBackend):
    """LLM backend that shells out to the `claude` CLI.

    Calls are billed against the user's Claude Pro/Max subscription via
    OAuth credentials stored in CLAUDE_CONFIG_DIR.
    """

    def __init__(
        self,
        model: str = "sonnet",
        max_retries: int = 3,
        claude_config_dir: Optional[str] = None,
        timeout: int = 300,
        extra_args: Optional[list] = None,
        # Accepted and silently ignored for compatibility with the factory:
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Args:
            model: Claude alias ("haiku", "sonnet", "opus") or full model id
            max_retries: retry attempts for transient subprocess failures
            claude_config_dir: override CLAUDE_CONFIG_DIR (where OAuth tokens live)
            timeout: per-call subprocess timeout, seconds
            extra_args: additional CLI args injected into every invocation
            api_key: IGNORED. Present only for factory compatibility.
            base_url: IGNORED. Present for config compatibility.
        """
        self.model = model
        self.max_retries = max_retries
        self.claude_config_dir = (
            claude_config_dir
            or os.environ.get("CLAUDE_CONFIG_DIR")
            or DEFAULT_CLAUDE_CONFIG_DIR
        )
        self.timeout = timeout
        self.extra_args = list(extra_args or [])

        # Locate claude executable
        self._claude_path = self._find_claude_executable()
        if not self._claude_path:
            raise RuntimeError(
                "claude CLI not found. Install Claude Code and ensure "
                "`claude` is on PATH. See: https://docs.anthropic.com/en/docs/claude-code"
            )

    # ---------------- Public API ----------------

    def complete(
        self,
        messages,
        tools=None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Execute one completion against Claude Code."""
        system_prompt, prompt_text = self._serialize_messages(messages)

        # Build CLI argv
        argv = [
            self._claude_path,
            "-p",
            "--output-format", "json",
            "--no-session-persistence",
            "--disable-slash-commands",
            "--tools", "",
            "--model", self.model,
        ]
        if system_prompt:
            argv += ["--system-prompt", system_prompt]

        # Tool-use emulation via prompt injection
        if tools:
            prompt_text = self._inject_tool_context(prompt_text, tools)

        argv += self.extra_args
        argv.append(prompt_text)

        # Subprocess environment
        env = os.environ.copy()
        env["CLAUDE_CONFIG_DIR"] = self.claude_config_dir
        env.pop("ANTHROPIC_API_KEY", None)

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                result = subprocess.run(
                    argv,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    env=env,
                )
                if result.returncode != 0:
                    # Try to recover valid completion from stdout
                    if result.stdout:
                        try:
                            data = json.loads(result.stdout)
                            if isinstance(data, dict) and not data.get("is_error"):
                                return self._parse_output(result.stdout, tools)
                        except (json.JSONDecodeError, ValueError):
                            pass
                    raise RuntimeError(
                        f"claude CLI exit={result.returncode}: "
                        f"stderr={result.stderr[:800]!r} "
                        f"stdout={result.stdout[:800]!r}"
                    )
                return self._parse_output(result.stdout, tools)
            except subprocess.TimeoutExpired as e:
                last_error = e
            except Exception as e:
                last_error = e
            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)

        raise RuntimeError(
            f"ClaudeCodeBackend failed after {self.max_retries} retries: "
            f"{last_error}"
        )

    def get_model_id(self) -> str:
        return f"claude-code:{self.model}"

    @property
    def supports_tool_use(self) -> bool:
        return True

    # ---------------- Helpers ----------------

    @staticmethod
    def _find_claude_executable() -> Optional[str]:
        """Locate claude on PATH."""
        for p in os.environ.get("PATH", "").split(os.pathsep):
            if not p:
                continue
            candidate = os.path.join(p, "claude")
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
        return None

    @staticmethod
    def _serialize_messages(messages):
        """Flatten multi-turn history into (system_prompt, user_prompt)."""
        system_parts = []
        turns = []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
            elif m.role == "user":
                turns.append(("USER", m.content))
            elif m.role == "assistant":
                if m.tool_calls:
                    tc_repr = json.dumps(m.tool_calls, indent=2)
                    content = m.content or ""
                    turns.append(
                        ("ASSISTANT", f"{content}\n\n[tool calls issued]:\n{tc_repr}")
                    )
                else:
                    turns.append(("ASSISTANT", m.content or ""))
            elif m.role == "tool_result":
                turns.append(
                    ("TOOL_RESULT", f"(for tool_call_id={m.tool_call_id})\n{m.content}")
                )
            else:
                turns.append((m.role.upper(), m.content or ""))

        system_prompt = "\n\n".join(system_parts) if system_parts else None

        if len(turns) == 1 and turns[0][0] == "USER":
            prompt = turns[0][1]
        else:
            parts = [f"{role}:\n{content}" for role, content in turns]
            prompt = (
                "Conversation so far:\n\n"
                + "\n\n---\n\n".join(parts)
                + "\n\n---\n\nPlease produce your next assistant response."
            )

        return system_prompt, prompt

    @staticmethod
    def _inject_tool_context(prompt: str, tools) -> str:
        """Prepend tool descriptions and calling instructions to the prompt."""
        lines = ["# Available tools", ""]
        for t in tools:
            lines.append(f"## {t['name']}")
            desc = t.get("description", "(no description provided)")
            lines.append(desc)
            schema_json = json.dumps(t.get("input_schema", {}), indent=2)
            lines.append("Input schema:")
            lines.append("```json")
            lines.append(schema_json)
            lines.append("```")
            lines.append("")
        lines += [
            "# Instructions",
            "",
            "You must respond with a single JSON object matching:",
            '  `{ "tool_name": <name of exactly one tool above>,',
            '     "arguments": <object matching that tool\'s input_schema> }`',
            "",
            "Do not include any explanatory prose outside the JSON object.",
            "",
            "---",
            "",
        ]
        return "\n".join(lines) + prompt

    def _parse_output(self, stdout: str, tools) -> LLMResponse:
        """Parse `claude -p --output-format json` stdout into an LLMResponse."""
        content = ""
        usage_in = 0
        usage_out = 0
        stop_reason = "end_turn"
        structured: Optional[dict] = None

        try:
            data = json.loads(stdout)
            content = data.get("result") or ""
            structured_raw = data.get("structured_output")
            if isinstance(structured_raw, dict):
                structured = structured_raw

            usage = data.get("usage") or {}
            if isinstance(usage, dict):
                try:
                    usage_in = int(usage.get("input_tokens") or 0)
                    usage_out = int(usage.get("output_tokens") or 0)
                except (TypeError, ValueError):
                    pass

            stop_reason = data.get("stop_reason") or "end_turn"

            if data.get("is_error"):
                err_msg = data.get("result") or data.get("error") or "claude reported is_error=true"
                raise RuntimeError(f"claude CLI error: {err_msg}")
        except json.JSONDecodeError:
            content = stdout.strip()
        except (ValueError, TypeError):
            pass

        # Tool-call extraction
        tool_calls = None
        if tools:
            if structured is not None:
                tool_calls = self._tool_call_from_structured(structured, tools)
            if tool_calls is None and content:
                tool_calls = self._extract_tool_call(content, tools)
            if tool_calls:
                stop_reason = "tool_use"

        return LLMResponse(
            content="" if tool_calls else content,
            tool_calls=tool_calls,
            usage={"input_tokens": usage_in, "output_tokens": usage_out},
            stop_reason=stop_reason,
        )

    @staticmethod
    def _tool_call_from_structured(structured: dict, tools):
        """Convert structured_output dispatch object into tool_calls list."""
        tool_name = structured.get("tool_name")
        arguments = structured.get("arguments")
        if not isinstance(tool_name, str):
            return None
        if not isinstance(arguments, dict):
            return None
        valid_names = {t["name"] for t in tools}
        if tool_name not in valid_names:
            return None
        return [
            {
                "id": f"cc_{tool_name}",
                "name": tool_name,
                "arguments": arguments,
            }
        ]

    @staticmethod
    def _extract_tool_call(content: str, tools):
        """Parse content as JSON matching the dispatch schema."""
        try:
            text = content.strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                newline = text.find("\n")
                if newline != -1:
                    text = text[newline + 1:]
                if text.rstrip().endswith("```"):
                    text = text.rstrip()[:-3].rstrip()
            obj = json.loads(text)
            tool_name = obj.get("tool_name")
            arguments = obj.get("arguments")
            if not isinstance(tool_name, str):
                return None
            if arguments is None or not isinstance(arguments, dict):
                return None
            valid_names = {t["name"] for t in tools}
            if tool_name not in valid_names:
                return None
            return [
                {
                    "id": f"claudecode_{tool_name}_{abs(hash(text)) % 10**8}",
                    "name": tool_name,
                    "arguments": arguments,
                }
            ]
        except (json.JSONDecodeError, AttributeError, KeyError, TypeError):
            return None

    def __repr__(self):
        return f"ClaudeCodeBackend(model={self.model!r}, timeout={self.timeout})"
