# AutoArchitect — CLAUDE.md

## Project Overview

AutoArchitect is a Flask-based self-learning Neural Architecture Search (NAS) system. It accepts plain English problem descriptions and automatically classifies, trains, and deploys neural networks using a multi-agent pipeline with a self-improving brain.

Entry point: `autoarchitect/app.py` (runs on port 5000).

## Repository Layout

```
autoarchitect/
├── app.py                        # Flask server — all routes
├── api/
│   ├── orchestrator.py           # Main controller (AutoArchitectOrchestrator)
│   ├── nas_engine.py             # DARTS NAS: DARTSNet, MixedOp, DARTSCell
│   ├── analyzer.py               # BERT problem domain classifier
│   ├── self_trainer.py           # Auto self-training pipeline
│   ├── transfer_trainer.py       # ResNet18 transfer learning
│   ├── auto_trainer.py           # Base model selection + YOLO detection
│   ├── cache_manager.py          # Problem→result cache (JSON files)
│   ├── dataset_fetcher.py        # HuggingFace dataset loading
│   ├── dataset_manager.py        # Dataset management utilities
│   ├── data_uploader.py          # User-uploaded labeled data handling
│   ├── workflow_engine.py        # Rule-based workflow builder (fallback)
│   ├── agents/
│   │   ├── base_agent.py         # Abstract base class for all agents
│   │   ├── image_agent.py        # Vision domain NAS agent
│   │   ├── text_agent.py         # NLP domain NAS agent
│   │   ├── medical_agent.py      # Medical imaging agent
│   │   ├── security_agent.py     # Security/fraud detection agent
│   │   ├── fusion_agent.py       # Merges multi-agent results
│   │   ├── evaluator_agent.py    # Scores architecture quality
│   │   ├── agent_network.py      # Agent network runner
│   │   ├── agent_runtime.py      # Agent execution runtime
│   │   ├── agent_factory.py      # Creates agents from trained models
│   │   └── dynamic_agent.py      # Dynamically created agents
│   └── brain/
│       ├── workflow_generator.py # Brain — generates optimal workflows
│       ├── meta_learner.py       # Learns from every solved problem
│       ├── topology_designer.py  # Designs multi-agent network topologies
│       ├── self_evaluator.py     # Scores brain's own output
│       ├── web_researcher.py     # Searches internet for best approach
│       ├── data_discovery_engine.py  # Multi-source dataset discovery
│       ├── network_zip_generator.py  # Generates deployable agent ZIPs
│       ├── agent_generator.py    # Generates agent Python code
│       ├── output_generator.py   # Human-readable result formatting
│       ├── performance_tracker.py# Tracks per-strategy accuracy history
│       └── strategy_library.py   # Stores known strategies per domain
├── brain_data/                   # Persisted brain state (JSON)
├── datasets/                     # Downloaded + cached datasets
├── models/                       # Trained model weights (.pth)
├── user_data/                    # User-uploaded data per problem
├── static/                       # Frontend CSS + JS
└── templates/index.html          # Single-page UI
```

## Core Pipeline

**Problem → Result flow (orchestrator.py `solve()`):**

1. LLM check — if the problem is text generation, route to Groq/Llama 3
2. BERT classifier → domain (`image` / `text` / `medical` / `security`)
3. Domain correction via `_correct_domain()` heuristics
4. Cache lookup — if found, return instantly
5. Similar-problem check — reuse close match
6. Brain (`WorkflowGenerator`) generates optimal workflow (single or multi-agent)
7. Web researcher finds best model/dataset for this problem
8. Agents run NAS + self-training
9. FusionAgent merges multi-agent results
10. EvaluatorAgent scores architecture
11. TopologyDesigner designs agent network topology
12. SelfEvaluator scores the generated ZIP
13. Brain learns from result for next time

## Key Design Decisions

- **DARTS NAS** (`DARTSNet`) uses continuous relaxation over 5 ops: skip, conv3x3, conv5x5, maxpool, avgpool. Architecture weights are separate from network weights.
- **Transfer learning** (ResNet18) is preferred for image/medical domains; DARTS is used for text/security.
- **Cache is truth** — every solved problem is saved to `cache/`. On cache hit the result is returned immediately without re-training.
- **Brain threshold** — MetaLearner only overrides BERT domain when ≥85% confident AND ≥60% historical accuracy. Lower thresholds caused wrong agent selection.
- **Agent lazy loading** — agents are instantiated on demand and unloaded after use to save memory (`_wake_agent` / `_sleep_agent`).
- **Self-evaluation feedback loop** — `SelfEvaluator` scores the generated ZIP and feeds that score back to `TopologyDesigner`, so topology selection improves over time.

## Environment Variables

```
GROQ_API_KEY=   # Free at console.groq.com — used for Llama 3 LLM calls
```

## Running Locally

```bash
cd autoarchitect
pip install torch torchvision flask flask-cors transformers datasets \
            huggingface-hub requests python-dotenv Pillow scikit-learn
python app.py   # http://localhost:5000
```

## Persistent Data Directories

| Directory | Contents |
|---|---|
| `brain_data/` | Strategies, meta-learner examples, topology history, eval history |
| `datasets/hf_cache/` | HuggingFace downloaded datasets |
| `datasets/discovery_cache/` | Data discovery query cache |
| `models/trained/` | Trained model weights + class label files |
| `user_data/` | User-uploaded labeled datasets per problem slug |
| `cache/` | Problem → architecture/accuracy cache entries |
