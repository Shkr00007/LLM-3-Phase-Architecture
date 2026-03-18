# LLM 3-Phase Architecture

A Python 3.11+ multi-agent system that uses a remote Ollama server, SQLite persistence, and a local subject-predicate-object graph.

## Agents
- **Repository Agent**: ingestion, chunking, embedding, graph extraction, hybrid retrieval, and context pruning.
- **Diplomat Agent**: user-facing planner using a grounded ReAct-style JSON draft.
- **Judge Agent**: LLM-as-a-Judge verifier that can force retries with targeted feedback.
- **Conductor**: strict JSON handoff orchestrator.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python main.py check
python main.py ingest notes.txt
python main.py query "What do we know about the customer roadmap?"
```

## Persistence
All durable memory is stored in `database/memory.db`, including chunk embeddings, conversation turns, graph triplets, and mid-term summaries.
