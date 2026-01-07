
import sys
import os
import time
import random
import numpy as np

# Add src to path
sys.path.append(os.path.abspath("src"))

from swarm_hybrid_evolution import HybridSwarmController
from swarm_brain import SwarmBrain

# 1. PREPARE 250 WORDS
# Mix of thematic clusters to test resonance and category inhibition
base_words = [
    "rower", "samochód", "motocykl", "pociąg", "samolot", "statek", "hulajnoga", "traktor",
    "matematyka", "fizyka", "biologia", "chemia", "astronomia", "geografia", "historia", "logika",
    "pies", "kot", "chomik", "papuga", "ryba", "koń", "krowa", "świnia", "owca", "wilk", "leśny",
    "programowanie", "python", "javascript", "algorytm", "komputer", "serwer", "baza", "dane",
    "chleb", "masło", "ser", "szynka", "pomidor", "ogórek", "cebula", "czosnek", "zupa", "obiad",
    "dom", "mieszkanie", "okno", "drzwi", "ściana", "dach", "podłoga", "sufit", "pokój", "kuchnia"
]

# Expand to 250 by adding variants and random-like strings if needed,
# but let's just multiply base and add suffixes for semantic closeness
expanded_words = []
for i in range(5):
    for w in base_words:
        expanded_words.append(f"{w}_{i}")

# Ensure exactly 250 or more
while len(expanded_words) < 250:
    expanded_words.append(f"pojęcie_{len(expanded_words)}")

random.shuffle(expanded_words)
test_words = expanded_words[:250]

print(f"✅ Prepared {len(test_words)} words for stress test.")

# 2. INITIALIZE SYSTEM
# Clear old memory to avoid vector size mismatch (10 vs 16)
if os.path.exists("swarm_ltm_store.pkl"):
    os.remove("swarm_ltm_store.pkl")
    print("🗑️ Cleared old memory file.")

# Start with 50 agents as per user requirement "minimalna liczba agentów = 50" (A4)
print("🚀 Initializing Hybrid Swarm with 50 agents...")
swarm = HybridSwarmController(size=50)

# Optional: Initialize Brain to see if it handles LTM side-by-side
brain = SwarmBrain()

# 3. RUN STRESS TEST (100 CYCLES)
print(f"🧬 Starting Stress Test: 100 Cycles | 250 Words")
print("-" * 60)

start_time = time.time()
metrics_history = []

for c in range(100):
    # Pick a random word from the 250 set
    current_word = test_words[c % len(test_words)]

    # Run 1 cycle of experiment
    swarm.run_experiment(current_word, cycles=1, brain=brain)

    # B2/C2. Update Brain LTM and check resonance
    # Swarm brain needs the engine (codec) to encode words
    brain.update_memory(current_word, semantic_engine=swarm.codec)

    # Perform pruning periodically (every 20 cycles)
    if c % 20 == 0:
        brain.prune_memory()

    # Monitor regression stats (Updated: Pass brain)
    stats = swarm.monitor_regression(brain=brain)
    metrics_history.append(stats)

    if c % 10 == 0:
        print(f"--- [Cycle {c}] Pop: {stats['population']} | Var: {stats['variance']:.5f} | UnrelatedSim: {stats['unrelated_sim']:.4f} ---")

end_time = time.time()
print("-" * 60)
print(f"🏁 Stress Test Completed in {end_time - start_time:.2f}s")

# 4. FINAL VERIFICATION
final_stats = swarm.monitor_regression()
print("\n📊 FINAL SYSTEM STATE:")
print(f"  - Population: {final_stats['population']}")
print(f"  - Average Age: {final_stats['avg_age']:.2f}")
print(f"  - Thought Variance: {final_stats['variance']:.6f}")
print(f"  - Unrelated Similarity: {final_stats['unrelated_sim']:.4f}")
print(f"  - Genome Drift: {final_stats['genome_drift']:.6f}")

if final_stats['population'] >= 50 and final_stats['unrelated_sim'] < 0.6:
    print("\n✅ SYSTEM STABLE: Diversification preserved, Scaling active.")
else:
    print("\n⚠️ WARNING: System might be converging too much or failed to grow.")
