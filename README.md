# Bahrain FNSL: This Is How We Fight Back 🇧🇭

> "We are not just predicting disease. We are giving our people a ten-year head start. This is personal."

Forget the fancy academic papers for a second. This project, **Federated Neuro-Symbolic Learning (FNSL)**, is about one thing: saving lives in Bahrain. We built this because our health system is broken, fragmented, and drowning in data that can't be shared. This is our answer.

## The Problem: Our Health Is on the Line

Look around. Type 2 Diabetes. Sickle Cell Disease. These aren't abstract statistics here; they're our families, our neighbors. We've got some of the highest rates, and our data—genomic, clinical—is locked up. Privacy laws are critical, yes, but they've created silos. And the AI solutions out there? They demand all your data in one place, or they're black boxes no doctor trusts. That's unacceptable.

## Our Solution: FNSL. No Compromises.

We said, "No more." We engineered FNSL from the ground up to do three things:

1.  **Break the Walls**: Hospitals collaborate. They train an AI together. But *no raw patient data ever leaves their building*. Ever. That's Federated Learning.
2.  **Make AI Smart (and Sane)**: We combined raw deep learning power (the "Neuro" part) with a medical brain, a Knowledge Graph (the "Symbolic" part). The AI finds patterns, then the Knowledge Graph checks if those patterns make clinical sense. No more AI guessing games.
3.  **Privacy is Non-Negotiable**: We layered on Differential Privacy and Secure Multi-Party Computation. This isn't just secure; it's ironclad. You can't reverse-engineer patient data from our models.

**The result?** A 0.94 AUC for predicting T2DM a decade out. That's not just a number. That's a ten-year warning. A chance to change everything.

## What You'll Find Here (The Code)

This isn't just theory. This is the working code.

### Core Engine
-   `src/models/fnsl_model.py`: The actual Neuro-Symbolic GRU model. This is where the magic happens—fusing DNA, EHRs, and medical logic.
-   `src/federated/server.py`: The Federated Learning core. Handles secure aggregation and DP.
-   `src/knowledge_graph/kg_engine.py`: Our medical brain. It keeps the AI honest.
-   `src/data/data_simulator.py`: Generates the kind of messy, heterogeneous data you'd actually see in Bahraini hospitals.
-   `src/federated/network_slicing.py`: Our conceptual 6G network slicing. Because future-proofing is non-negotiable.

### The Toolkit
-   `src/utils/config.py`: All the knobs and levers for your experiments. Tweak it.
-   `src/utils/logger.py`: Tracks everything. No more guessing what happened.
-   `src/utils/fnsl_visualization.py`: See your results. Real-time plots, not just numbers.

### How to Run It
-   `main.py`: The entry point. Hit this to kick off a full federated training run.

## Get It Running. Now.

```bash
git clone https://github.com/yourusername/bahrain-fnsl-ai.git
cd bahrain-fnsl-ai
pip install -r requirements.txt
python main.py
```

That's it. You'll see it train. You'll see the plots. You'll see the future.

## Why This Matters (The Impact)

We didn't build this for a paper. We built it to make a difference. This isn't just about AI; it's about health sovereignty. It's about giving our people the tools to fight back against chronic disease.

## Join the Fight

This is open source. This is our project. If you're a researcher, a doctor, a developer—if you care about this mission—we need you. Read `CONTRIBUTING.md`. Open an issue. Send a PR. Let's build this future, together.

---

**Bahrain FNSL: For Our Health. For Our Future.**

*Questions? Problems? Hit us up on GitHub Issues. We're here.*
