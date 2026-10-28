# Self-Correcting Agentic Development Harness

This harness implements a closed-loop verification state machine using LangGraph. It compiles code hypotheses, tests them deterministically using pytest in an isolated subprocess, and feeds error diffs back into the LLM or CLI agent until all tests pass.

## Supported Providers
1. **OpenAI API** (`gpt-4o`) via `langchain-openai`
2. **Google Gemini API** (`gemini-2.5-flash`) via `langchain-google-genai`
3. **Claude Code CLI** via headless subprocess (`claude -p -r`)

## Setup
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

## Running the Harness
```bash
# Run with Claude Code CLI
python main.py --provider claude-code

# Run with Gemini
python main.py --provider gemini

# Run with OpenAI
python main.py --provider openai
```
