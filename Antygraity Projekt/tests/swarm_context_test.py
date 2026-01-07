
import numpy as np
import sqlite3
import random
import time
from swarm_sentence_processor import setup_ltm, LorenzAgent, text_to_vector

# ==========================================
# EXTENDED LTM SETUP
# ==========================================
def setup_context_ltm():
    conn = sqlite3.connect("swarm_memory.db")
    cursor = conn.cursor()
    # Ensure table exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS context_memories (
            keyword TEXT PRIMARY KEY,
            vector BLOB
        )
    """)

    # Define strong attractors (The "Hidden Contexts")
    # We create a specific vector for "BIKE_CONTEXT"
    # To simulate "Relatedness", we'll make related words structurally similar to this vector

    # Base Vector for Bike Context
    bike_base = np.zeros(16, dtype=np.float32)
    bike_base[0:4] = 0.8 # Strong activation in first dimension

    seed_data = [
        ("CONTEXT_BIKE", bike_base.tobytes()),
        ("CONTEXT_WORK", (np.zeros(16, dtype=np.float32) + 0.1).tobytes()) # Different pattern
    ]

    # Upsert logic (replace if exists)
    for key, blob in seed_data:
        cursor.execute("REPLACE INTO context_memories VALUES (?, ?)", (key, blob))

    conn.commit()
    return conn, bike_base

# ==========================================
# CONTEXT TEST CONTROLLER
# ==========================================
class ContextTestController:
    def __init__(self, num_agents=15):
        self.conn, self.target_context = setup_context_ltm()
        self.agents = [LorenzAgent(i, self.conn) for i in range(num_agents)]

    def get_swarm_consensus(self):
        # Average state of Output Layer (Agents 10-14)
        output_vec = np.zeros(16, dtype=np.float32)
        for i in range(10, 15):
            output_vec += self.agents[i].vector_state
        return output_vec / 5.0

    def process_sequence(self, sentences):
        print(f"\n{'='*60}")
        print(f"🚴 STARTING CONTEXT HYPOTHESIS TEST")
        print(f"Goal: Check if swarm drifts towards 'CONTEXT_BIKE' state.")
        print(f"{'='*60}")

        # We manually craft input vectors to be "somewhat similar" to bike_base
        # to simulate semantic relatedness without a real BERT model.
        # Real word: "Rower" -> text_to_vector("Rower")
        # But here we urge the simulation:

        for s_idx, sentence in enumerate(sentences):
            print(f"\n📝 SENTENCE {s_idx+1}: '{sentence}'")
            tokens = sentence.split()

            for t_idx, token in enumerate(tokens):
                # encode
                vec = text_to_vector(token)

                # HACK: If token implies bike, nudge vector slightly towards bike_base
                # This simulates semantic embedding similarity
                if token in ["rower", "trasa", "łańcuch", "kilometrów"]:
                    # STRONGER NUDGE (Was 0.3)
                    vec = 0.2 * vec + 0.8 * self.target_context
                    print(f"   >>> 🚴 DETECTED KEYWORD '{token}' -> STRONG RESONANCE APPLIED")

                # Reduce Chaos via Damping in Layers
                # L1
                l1_out = np.mean([a.process_input(vec) for a in self.agents[:5]], axis=0) * 0.95
                # L2
                l2_out = np.mean([a.process_input(l1_out) for a in self.agents[5:10]], axis=0)
                # L3
                l3_out = np.mean([a.process_input(l2_out) for a in self.agents[10:]], axis=0)

                # Measure Distance to Target Context
                consensus = self.get_swarm_consensus()
                dist = np.linalg.norm(consensus - self.target_context)

                # Bar chart for distance
                bar = "#" * int((2.0 - dist) * 20)
                print(f"   Token '{token}' -> Swarm Distance to BIKE_CONTEXT: {dist:.4f} {bar}")

                # Dynamics Settle
                for _ in range(3):
                    for a in self.agents: a.step_lorenz()

        print(f"\n{'='*60}")
        print("🏁 SEQUENCE COMPLETE")

if __name__ == "__main__":
    test = ContextTestController()

    story = [
        "Lubię aktywnie spędzać czas",
        "Wczoraj zrobiłem 30 kilometrów",
        "Trasa przez las była super",
        "Muszę nasmarować łańcuch w rowerze"
    ]

    test.process_sequence(story)
