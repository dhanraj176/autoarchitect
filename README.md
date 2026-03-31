# AutoArchitect — Self-Learning Multi-Agent Neural Architecture Search

AutoArchitect is an AI-powered system that automatically designs, trains, and deploys neural networks from plain English problem descriptions. It combines Differentiable Architecture Search (DARTS NAS) with a multi-agent pipeline and a self-learning brain.

> **Oakland Research Showcase 2026** — AI beats human baseline by +22.33% at $0 cost.

---

## What It Does

You describe a problem in plain English. AutoArchitect:

1. Classifies the problem domain using BERT (image / text / medical / security)
2. Fetches the best matching dataset from HuggingFace, Kaggle, Papers With Code, or OpenImages
3. Runs DARTS Neural Architecture Search to discover the optimal network structure
4. Trains the model using ResNet18 transfer learning (vision) or NAS (text/security)
5. Fuses results from multiple specialized agents if needed
6. Self-evaluates its own output and feeds the score back to improve future decisions
7. Lets you download the full trained agent network as a deployable ZIP

---

## Architecture

```
┌─────────────────────────────────────────┐
│              Flask API (app.py)          │
└────────────────┬────────────────────────┘
                 │
    ┌────────────▼────────────┐
    │  AutoArchitectOrchestrator  │
    │  (api/orchestrator.py)      │
    └──┬─────────┬────────────┘
       │         │
  ┌────▼───┐  ┌──▼──────────────────────────┐
  │  BERT  │  │         Brain System         │
  │Analyzer│  │  (api/brain/)                │
  └────────┘  │  WorkflowGenerator           │
              │  MetaLearner                 │
              │  TopologyDesigner            │
              │  SelfEvaluator               │
              │  WebResearcher               │
              │  NetworkZipGenerator         │
              └──────────┬──────────────────┘
                         │
            ┌────────────▼─────────────────┐
            │       Agent Network          │
            │  ImageAgent  TextAgent       │
            │  MedicalAgent SecurityAgent  │
            │  FusionAgent EvaluatorAgent  │
            └──────────────────────────────┘
```

### Key Modules

| Module | Purpose |
|---|---|
| [autoarchitect/app.py](autoarchitect/app.py) | Flask server — all REST API routes |
| [api/orchestrator.py](autoarchitect/api/orchestrator.py) | Main controller — problem → result pipeline |
| [api/nas_engine.py](autoarchitect/api/nas_engine.py) | DARTS NAS — DARTSNet, MixedOp, DARTSCell |
| [api/analyzer.py](autoarchitect/api/analyzer.py) | BERT problem classifier |
| [api/self_trainer.py](autoarchitect/api/self_trainer.py) | Auto training on fetched datasets |
| [api/transfer_trainer.py](autoarchitect/api/transfer_trainer.py) | ResNet18 transfer learning |
| [api/cache_manager.py](autoarchitect/api/cache_manager.py) | Result caching (2066x speedup on cache hit) |
| [api/dataset_fetcher.py](autoarchitect/api/dataset_fetcher.py) | HuggingFace dataset fetching |
| [api/brain/workflow_generator.py](autoarchitect/api/brain/workflow_generator.py) | Brain — generates optimal workflows |
| [api/brain/meta_learner.py](autoarchitect/api/brain/meta_learner.py) | Learns from every problem solved |
| [api/brain/topology_designer.py](autoarchitect/api/brain/topology_designer.py) | Designs multi-agent network topologies |
| [api/brain/self_evaluator.py](autoarchitect/api/brain/self_evaluator.py) | Scores the brain's own output |
| [api/brain/web_researcher.py](autoarchitect/api/brain/web_researcher.py) | Searches for best models/datasets per problem |
| [api/brain/data_discovery_engine.py](autoarchitect/api/brain/data_discovery_engine.py) | Multi-source dataset discovery |
| [api/brain/network_zip_generator.py](autoarchitect/api/brain/network_zip_generator.py) | Generates deployable agent network ZIPs |
| [api/agents/](autoarchitect/api/agents/) | Specialized domain agents |

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web UI |
| `/api/analyze` | POST | Classify problem domain with BERT |
| `/api/search` | POST | Run NAS search (with cache) |
| `/api/orchestrate` | POST | Full pipeline — problem → trained model |
| `/api/self-train` | POST | Trigger self-training on a problem |
| `/api/train` | POST | Train with auto-selected base model |
| `/api/detect` | POST | YOLO object detection on an image |
| `/api/predict` | POST | Inference on a trained NAS model |
| `/api/predict-user` | POST | Inference on a user-uploaded trained model |
| `/api/upload-data` | POST | Upload labeled data and train a custom model |
| `/api/download/multi-nas` | POST | Download full NAS package as ZIP |
| `/api/download/network` | POST | Download generated agent network as ZIP |
| `/api/topology/preview` | POST | Preview agent topology for a problem |
| `/api/brain/status` | GET | Brain learning stats |
| `/api/brain/eval-stats` | GET | Self-evaluator statistics |
| `/api/cache/stats` | GET | Cache statistics |

---

## Setup

### 1. Install dependencies

```bash
pip install torch torchvision flask flask-cors transformers datasets \
            huggingface-hub requests python-dotenv Pillow scikit-learn
```

### 2. Configure environment

```bash
cp .env.template .env
# Edit .env — add your GROQ_API_KEY (free at console.groq.com)
```

### 3. Run the server

```bash
cd autoarchitect
python app.py
# Open http://localhost:5000
```

### 4. Run NAS from CLI

```bash
python run_nas.py --problem "detect potholes in road images"
python run_nas.py --problem "classify spam messages" --epochs 5
```

---

## Performance

| Metric | Human Baseline | AutoArchitect |
|---|---|---|
| Accuracy | 52.56% | 74.89% |
| Parameters | 1.6M | 105K |
| Cache speedup | 1x | 2066x |
| Cost | $$$ | $0 |

---

## Tech Stack

- **PyTorch** — DARTS NAS, ResNet18 transfer learning
- **Transformers / BERT** — problem domain classification
- **HuggingFace Datasets** — free dataset discovery and loading
- **Groq API (Llama 3.1)** — LLM for text generation tasks
- **Flask** — REST API server
- **scikit-learn** — meta-learner utilities
