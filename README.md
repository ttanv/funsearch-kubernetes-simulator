# FunSearch-Driven Kubernetes Scheduler Optimization


## Overview

This project demonstrates automated discovery of Kubernetes scheduling policies using Google's FunSearch algorithm. By combining discrete-event simulation with LLM-powered code evolution, we systematically discover scheduling strategies that outperform traditional approaches on real datacenter workloads.



## Key Innovation

Traditional schedulers use hand-crafted heuristics. We use FunSearch to automatically discover scheduling functions that optimize across multiple objectives:

- **High-Speed Simulation**: Order-of-seconds evaluation vs. order-of-minutes for existing simulators
- **Resource Utilization**: CPU, memory, and GPU efficiency
- **Fragmentation Minimization**: Balanced resource allocation patterns  
- **GPU-Aware Placement**: Individual GPU memory tracking and load balancing

## AI Discovery Results


###  Results

After ~2,000 generations of evolution, FunSearch discovered scheduling policies that **significantly outperform classical algorithms**:

| Algorithm | Policy Score | Scheduling Attempts | Repushes | GPU Fragmentation |
|-----------|--------------|-------------------|----------|-------------------|
| **🤖 FunSearch Best** | **0.7861** | **123** | **123** | **0.064** |
| 🧠 Best-Fit (classical) | 0.7855 | 175 | 175 | 0.038 |
| 📊 First-Fit (baseline) | 0.2934 | 18,819 | 18,819 | 0.066 |

*Tested on 8,152 pods across 16 nodes with real Alibaba datacenter traces (each run takes around 0.1s)*

**Key Insights from New Metrics:**
- **Lower repush events**: FunSearch achieves 30% fewer repushes than best-fit (123 vs 175)
- **Reduced scheduling attempts**: Indicates better initial placement decisions
- **Fragmentation trade-offs**: FunSearch balances efficiency with slightly higher GPU fragmentation than best-fit



## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Evaluate existing policies (original evaluator)
python tests/test_scheduler.py

# Evaluate with new repush-focused metrics (dev branch)
python tests/test_repush_scheduler.py

# View the code for the top 3 discovered policies
# See tests/test_scheduler.py for implementation details

# Configure API key in configs/llm_config.json
# Replace "API_KEY" with your OpenRouter API key

# Run FunSearch to discover new policies  
python funsearch/funsearch_integration.py
```

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FunSearch Evolution Loop                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐           │
│ │   Population    │    │   LLM Policy    │    │   Evaluation    │           │
│ │   Management    │───▶│   Generator     │───▶│   & Selection   │───┐       │
│ │                 │    │                 │    │                 │   │       │
│ │ • Elite policies│    │ • Code gen      │    │ • Fitness calc  │   │       │
│ │ • Mutations     │    │ • Safety checks │    │ • Ranking       │   │       │
│ └─────────────────┘    └─────────────────┘    └─────────────────┘   │       │
│          ▲                                                          │       │
│          └──────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Kubernetes Simulator Core                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐           │
│ │  Event Engine   │───▶│   Scheduler     │───▶│   Performance   │           │
│ │                 │    │   Interface     │    │   Evaluator     │           │
│ │ • Pod creation  │    │                 │    │                 │           │
│ │ • Pod deletion  │    │ • Policy exec   │    │ • CPU/Mem/GPU   │           │
│ │ • Time ordering │    │ • Resource check│    │ • Fragmentation │           │
│ │                 │    │                 │    │ • # of Repushes │           │
│ └─────────────────┘    └─────────────────┘    └─────────────────┘           │
│          ▲                       │                       ▲                  │
│          │              ┌─────────────────┐              │                  │
│          │              │  Cluster State  │              │                  │
│          │              │                 │              │                  │
│          │              │ • Nodes/GPUs    │              │                  │
│          └──────────────│ • Resource mgmt │──────────────┘                  │
│                         │ • Validation    │                                 │
│                         └─────────────────┘                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ▲
                                       │
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Data Pipeline                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐           │
│ │  OpenB Dataset  │───▶│  Trace Parser   │───▶│  Entity Models  │           │
│ │                 │    │                 │    │                 │           │
│ │ • Workload CSVs │    │ • CSV processing│    │ • Pod objects   │           │
│ │ • Node configs  │    │ • Data cleaning │    │ • Node objects  │           │
│ │ • GPU specs     │    │ • Validation    │    │ • Cluster state │           │
│ └─────────────────┘    └─────────────────┘    └─────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Core Components

- **`simulator/`** - High-performance discrete-event simulation engine
- **`funsearch/`** - LLM-powered evolutionary policy discovery  
- **`policies/discovered/`** - Evolved scheduling policies with performance scores
- **`benchmarks/traces/`** - Real workload data from OpenB dataset

---

*This project showcases the practical application of program search for infrastructure optimization - using AI to discover algorithms that human engineers might not intuitively design.*
