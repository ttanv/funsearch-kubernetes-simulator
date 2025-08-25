# FunSearch-Driven Kubernetes Scheduler Optimization

> *Using LLM-powered program search to discover novel scheduling algorithms for GPU-accelerated workloads*

**Note:** `main` branch contains the original demoed version, `dev` branch contains realism improvements and fixes. Uses time-stepped evaluator and focuses on makespan metrics.

## Overview

This project demonstrates automated discovery of Kubernetes scheduling policies using Google's FunSearch algorithm. By combining discrete-event simulation with LLM-powered code evolution, we systematically discover scheduling strategies that outperform traditional approaches on real datacenter workloads.



## Key Innovation

Traditional schedulers use hand-crafted heuristics. We use FunSearch to automatically discover scheduling functions that optimize across multiple objectives:

- **High-Speed Simulation**: Order-of-seconds evaluation vs. order-of-minutes for existing simulators
- **Resource Utilization**: CPU, memory, and GPU efficiency
- **Fragmentation Minimization**: Balanced resource allocation patterns  
- **GPU-Aware Placement**: Individual GPU memory tracking and load balancing

## AI Discovery Results

After ~2,000 generations of evolution, FunSearch discovered scheduling policies that **significantly outperform classical algorithms**:

### Performance Comparison
| Algorithm | CPU Util | Memory Util | GPU Util | Fragmentation |
|-----------|----------|-------------|----------|---------------|
| **🤖 FunSearch Best** | **45.9%** | **26.1%** | **73.4%** | **0.033** |
| 🧠 Best-Fit (classical) | 42.6% | 23.6% | 68.6% | 0.039 |
| 📊 First-Fit (baseline) | 43.4% | 24.2% | 69.7% | 0.065 |

*Tested on 8,152 pods across 16 nodes with real Alibaba datacenter traces (each run takes around 0.1s)*

**Key Improvements:**
- **+7.7% CPU utilization** and **+10.6% memory utilization** vs best classical algorithm
- **+7.0% GPU utilization** with **15% less fragmentation** 
- At datacenter scale, this translates to millions in hardware cost savings and reduced energy consumption

### Champion Policy Innovation
```python
# Resource balance detection - penalizes CPU/memory ratio imbalances
balance_factor = abs(node.cpu_milli_left / max(1, node.memory_mib_left) - pod.cpu_milli / max(1, pod.memory_mib))
score -= balance_factor * 0.5

# Fragmentation prevention - reduces GPU memory waste  
gpu_fragmentation_factor = free_gpu_millis % pod.gpu_milli
score -= gpu_fragmentation_factor * 0.2
```

**Key Discoveries:**
- **Resource Balance Detection**: Penalizes CPU/memory ratio imbalances  
- **Fragmentation Prevention**: Reduces GPU memory waste
- **Adaptive Node Selection**: Context-aware bonuses for high-capacity nodes

## New Evaluation Metrics (Dev Branch)

The dev branch introduces a **repush-focused evaluator** that tracks scheduling efficiency and wait times instead of just resource utilization. This provides more realistic datacenter scheduling metrics:

### Evaluator Comparison

| Metric | Original Evaluator | Repush Evaluator (Dev) |
|--------|-------------------|------------------------|
| **Focus** | Resource utilization over time | Scheduling efficiency & makespan |
| **Primary Metrics** | CPU/Memory/GPU utilization % | Scheduling attempts, repushes, peak nodes |
| **Fragmentation** | Time-weighted fragmentation | Event-based fragmentation tracking |
| **Scoring** | Utilization - fragmentation penalty | Success rate + efficiency - wait penalty |
| **Use Case** | Resource optimization | Realistic scheduler performance |

### Dev Branch Results

| Algorithm | Policy Score | Scheduling Attempts | Repushes | GPU Fragmentation |
|-----------|--------------|-------------------|----------|-------------------|
| **🤖 FunSearch Best** | **0.7861** | **123** | **123** | **0.064** |
| 🧠 Best-Fit (classical) | 0.7855 | 175 | 175 | 0.038 |
| 📊 First-Fit (baseline) | 0.2934 | 18,819 | 18,819 | 0.066 |

*Results show FunSearch policy leaves less pods in pending state, leading to fewer repushes*

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
