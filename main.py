import argparse
import logging
import os
import subprocess
import uuid
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from harness_state import HarnessState
from model_factory import get_model

load_dotenv()

log = logging.getLogger("harness")

# Headless `claude -p` cannot prompt for approval, so unapproved actions are denied.
# Grant only what the task needs: read files, create/edit the target. Bash is not
# allowed (the harness runs pytest itself), and the test suite is protected so the
# agent cannot make tests pass by editing them.
CLAUDE_PERMISSION_ARGS = [
    "--permission-mode", "acceptEdits",
    "--allowedTools", "Read,Write,Edit",
    "--disallowedTools", "Edit(test_suite.py),Write(test_suite.py)",
]


def run_claude_code_cli(state: HarnessState, session_name: str) -> tuple[bool, str]:
    """Executes Claude Code via subprocess using headless and session flags.

    Returns (ok, output). ok is False on timeout, a missing binary or a non-zero
    exit, so the caller can report the failure instead of testing a stale file.
    """
    target = state["target_file"]
    prev_errors = state["test_output"]
    iteration = state["iteration_count"]

    if iteration == 0:
        prompt = (
            f"Create the file '{target}' and implement a complete LRUCache class with "
            f"capacity, get(key), and put(key, value) in Python. "
            f"Ensure all standard operations run in O(1) time complexity."
        )
        cmd = [
            "claude",
            "--name", session_name,
            "-p", prompt,
            *CLAUDE_PERMISSION_ARGS,
        ]
    else:
        prompt = (
            f"Verification of '{target}' failed with this output:\n"
            f"----------------------------------------\n"
            f"{prev_errors}\n"
            f"----------------------------------------\n"
            f"Directly modify and fix '{target}' to resolve the test failures."
        )
        cmd = [
            "claude",
            "-r", session_name,
            "-p", prompt,
            *CLAUDE_PERMISSION_ARGS,
        ]

    log.debug("Claude CLI prompt:\n%s", prompt)
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        log.debug("Claude CLI exit code: %s", res.returncode)
        log.debug("Claude CLI stdout:\n%s", res.stdout)
        if res.stderr:
            log.debug("Claude CLI stderr:\n%s", res.stderr)
        if res.returncode == 0:
            return True, res.stdout
        return False, f"Claude Code CLI exited with code {res.returncode}:\n{res.stdout}\n{res.stderr}"
    except subprocess.TimeoutExpired:
        return False, "Claude Code CLI execution timed out after 120 seconds."
    except FileNotFoundError:
        return False, "Error: 'claude' CLI binary not found in system PATH."


def build_harness(provider: str):
    is_claude_cli = (provider == "claude-code")
    llm = None if is_claude_cli else get_model(provider)

    def generator_node(state: HarnessState) -> dict:
        iteration = state["iteration_count"]

        # Path A: Claude Code CLI
        if is_claude_cli:
            target = state["target_file"]
            session_name = state.get("session_id") or f"harness-{uuid.uuid4().hex[:8]}"

            # Move any leftover target aside so a previous run's file cannot
            # produce a false PASS if Claude fails to write anything.
            if iteration == 0 and os.path.exists(target):
                backup = f"{target}.bak"
                log.debug("Moving existing %s to %s", target, backup)
                os.replace(target, backup)

            ok, cli_output = run_claude_code_cli(state, session_name)

            generator_error = None
            if not ok:
                generator_error = cli_output
            elif not os.path.exists(target):
                generator_error = (
                    f"Claude Code CLI finished but did not create '{target}'.\n{cli_output}"
                )

            return {
                "iteration_count": iteration + 1,
                "session_id": session_name,
                "proposed_code": None,  # Claude Code writes directly to disk
                "generator_error": generator_error,
                "messages": [HumanMessage(content=f"Claude CLI executed: {cli_output[:300]}...")]
            }

        # Path B: Standard API Model
        target = state["target_file"]
        prev_errors = state["test_output"]
        passed = state["tests_passed"]

        system_prompt = (
            "You are an automated software engineer inside an execution harness. "
            "Write robust, production-quality Python code satisfying all requirements and tests. "
            "Return ONLY executable Python code inside standard markdown code fences (```python ... ```). "
            "Do not include conversational text or explanations outside the code block."
        )

        user_content = f"Target implementation file: {target}\n\n"
        if prev_errors and not passed:
            user_content += (
                "Your previous code attempt failed test execution with the following errors:\n"
                "----------------------------------------\n"
                f"{prev_errors}\n"
                "----------------------------------------\n"
                "Analyze the test failures, fix any logic errors, and output the complete corrected file.\n"
            )
        else:
            user_content += (
                "Write the complete implementation for an LRUCache class with capacity, "
                "get(key), and put(key, value) methods matching standard O(1) requirements."
            )

        log.debug("Generator prompt (iteration %s):\n%s", iteration, user_content)
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            *state["messages"],
            HumanMessage(content=user_content)
        ])

        raw_text = response.content
        log.debug("Raw model response:\n%s", raw_text)
        if "```python" in raw_text:
            code = raw_text.split("```python")[1].split("```")[0].strip()
        elif "```" in raw_text:
            code = raw_text.split("```")[1].split("```")[0].strip()
        else:
            code = raw_text.strip()

        return {
            "proposed_code": code,
            "iteration_count": iteration + 1,
            "messages": [response]
        }

    def executor_node(state: HarnessState) -> dict:
        target = state["target_file"]
        code = state["proposed_code"]

        # The generator could not produce a file; report that instead of testing
        # whatever happens to be on disk.
        generator_error = state.get("generator_error")
        if generator_error:
            log.debug("Skipping pytest, generator failed:\n%s", generator_error)
            return {"tests_passed": False, "test_output": generator_error.strip()}

        if code is not None:
            log.debug("Writing %d chars to %s", len(code), target)
            with open(target, "w", encoding="utf-8") as f:
                f.write(code)

        try:
            res = subprocess.run(
                ["pytest", "test_suite.py", "-q", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=30
            )
            passed = (res.returncode == 0)
            output = res.stdout if passed else f"{res.stdout}\n{res.stderr}"
        except subprocess.TimeoutExpired:
            passed = False
            output = "Execution timed out after 30 seconds."
        except Exception as e:
            passed = False
            output = f"Execution harness error: {str(e)}"

        log.debug("Pytest passed=%s, output:\n%s", passed, output)
        return {
            "tests_passed": passed,
            "test_output": output.strip()
        }

    def error_evaluator_node(state: HarnessState) -> dict:
        raw_output = state["test_output"] or ""
        lines = raw_output.splitlines()
        truncated = "\n".join(lines[-25:]) if len(lines) > 25 else raw_output
        return {"test_output": truncated}

    def routing_condition(state: HarnessState) -> str:
        if state["tests_passed"]:
            return "passed"
        if state["iteration_count"] >= state["max_iterations"]:
            return "exhausted"
        return "retry"

    builder = StateGraph(HarnessState)
    builder.add_node("generator", generator_node)
    builder.add_node("executor", executor_node)
    builder.add_node("evaluator", error_evaluator_node)

    builder.add_edge(START, "generator")
    builder.add_edge("generator", "executor")
    builder.add_edge("executor", "evaluator")

    builder.add_conditional_edges(
        "evaluator",
        routing_condition,
        {
            "passed": END,
            "exhausted": END,
            "retry": "generator"
        }
    )

    return builder.compile()


def main():
    parser = argparse.ArgumentParser(description="LangGraph AI Agentic Harness")
    parser.add_argument(
        "--provider",
        choices=["openai", "gemini", "claude-code"],
        default=os.getenv("LLM_PROVIDER", "openai"),
        help="LLM provider switch (openai, gemini, or claude-code)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=4,
        help="Maximum self-correction iteration attempts"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed output (prompts, model responses, test output)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Keep third-party libraries quiet; only the harness's own debug output is shown.
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "langsmith"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    print("\n==========================================")
    print(" Initializing Agentic Harness")
    print(f" Provider: {args.provider.upper()}")
    print(f" Max Iterations: {args.max_iterations}")
    print("==========================================\n")

    app = build_harness(provider=args.provider)

    target_file = "lru_cache.py"
    initial_state = {
        "messages": [
            HumanMessage(content="Implement an LRU cache with get and put in Python.")
        ],
        "target_file": target_file,
        "proposed_code": None,
        "test_output": None,
        "tests_passed": False,
        "iteration_count": 0,
        "max_iterations": args.max_iterations,
        "session_id": None,
        "generator_error": None
    }

    for event in app.stream(initial_state):
        for node_name, state_update in event.items():
            print(f"[Completed Step]: {node_name}")
            log.debug("State update from %s: %s", node_name, state_update)
            if "iteration_count" in state_update:
                print(f" -> Iteration: {state_update['iteration_count']}")
            if "tests_passed" in state_update:
                status = "PASSED" if state_update["tests_passed"] else "FAILED"
                print(f" -> Test Result: {status}")

    print("\n==========================================")
    print(" Execution Completed")
    print("==========================================")
    if os.path.exists(target_file):
        print(f"Target file '{target_file}' verified on disk.")


if __name__ == "__main__":
    main()
