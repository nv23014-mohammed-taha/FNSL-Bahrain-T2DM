# Getting Started: No Fluff, Just Code

Alright, let's get this running. You're here because you want to see Federated Neuro-Symbolic Learning (FNSL) actually work, not just read about it. This guide is straight to the point.

## What You Need

- Python 3.8+. Don't try this on ancient versions.
- `pip` or `conda`. You know the drill.
- At least 4GB of RAM. 8GB if you want to crank up the patient numbers.
- A GPU is nice, but not strictly required for the default simulation.

## The Setup

### 1. Grab the Code
```bash
git clone https://github.com/yourusername/bahrain-fnsl-ai.git
cd bahrain-fnsl-ai
```

### 2. Isolate It
Don't mess up your system Python. Use a virtual environment.
```bash
python -m venv venv
source venv/bin/activate  # Windows folks: venv\Scripts\activate
```

### 3. Install the Heavy Lifters
```bash
pip install -r requirements.txt
```
This pulls in PyTorch, NumPy, Matplotlib, and the rest of the gang.

## Firing It Up

### The 5-Minute Test Drive
Just run the main script. It's pre-configured to do a full federated run with synthetic data.
```bash
python main.py
```

**What's happening under the hood?**
1.  It spins up a fake Knowledge Graph (our "medical brain").
2.  It generates a bunch of synthetic patients across 10 simulated hospitals. This data is messy on purpose (heterogeneous).
3.  It kicks off 50 rounds of Federated Learning.
4.  Watch the console. You'll see the global model updating and the local losses dropping.

### Where Did My Stuff Go?

-   **The Logs**: Check `logs/fnsl_experiments/`. Every metric, every config tweak is saved there in JSONL format.
-   **The Pictures**: Look in `results/fnsl_visualizations/`. You'll find plots showing how the AUC climbed and the loss fell.
-   **The Brain**: The final, trained global model is sitting in `checkpoints/fnsl_global_model.pth`.

## Tinkering Under the Hood

You didn't come here to just run defaults. Open up `src/utils/config.py`. This is your control panel.

```python
# Want more patients? Change this.
NUM_PATIENTS_TOTAL = 10000

# Want to see what happens when hospitals have wildly different data? Crank this up (0 to 1).
HETEROGENEITY_FACTOR = 0.7

# Need more training rounds?
FEDERATED_ROUNDS = 50

# Want to test the privacy limits? Adjust the noise.
DIFFERENTIAL_PRIVACY_NOISE_MULTIPLIER = 0.001
```
Change a number, save, run `main.py` again. See what breaks or what improves.

## How the Code is Actually Structured

If you're going to hack on this, you need to know where things live.

### The Core Engine (`src/models/fnsl_model.py`)
This is the beast. It's got the GRU for chewing through time-series EHR data, the linear layers for the SNP data, and the `NeuroSymbolicLayer` that forces the neural net to listen to the Knowledge Graph.

### The Network (`src/federated/server.py`)
This handles the Federated Averaging (FedAvg). It's where the `FederatedServer` aggregates updates and adds Differential Privacy noise. The `HospitalClient` class is also here—that's what runs the local training loop on the "sensitive" data.

### The Brain (`src/knowledge_graph/kg_engine.py`)
Right now, it's a simulated Knowledge Graph. In a real deployment, this hooks into a massive Neo4j database of medical facts.

### The Fake Data (`src/data/data_simulator.py`)
Because we can't ship real Bahraini health data on GitHub, this script generates synthetic SNPs and EHRs that mimic the statistical properties we care about (like T2DM prevalence).

## When Things Break (Troubleshooting)

-   **OOM (Out of Memory)**: You got greedy. Lower `BATCH_SIZE` or `NUM_PATIENTS_TOTAL` in `config.py`.
-   **It's Too Slow**: You're probably running on CPU with massive patient numbers. Either use a smaller dataset for testing or get a machine with a CUDA-enabled GPU.
-   **No Plots**: Make sure the script has permission to write to the `results/` directory.

## What's Next?

Read the actual paper in `docs/RESEARCH_PAPER.md`. It explains *why* we built this crazy architecture. Then, start ripping the code apart and making it better.

We need this to work in the real world. Let's get to it.
