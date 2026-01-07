
import numpy as np
import sqlite3
import random
import time

# ==========================================
# 1. DATABASE & MEMORY (LTM)
# ==========================================
def setup_ltm():
    conn = sqlite3.connect("swarm_memory.db")
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS context_memories")
    cursor.execute("""
        CREATE TABLE context_memories (
            keyword TEXT PRIMARY KEY,
            vector BLOB
        )
    """)
    # Seed known context (Marcin, Rower)
    # We use random vectors as "meaning" placeholders
    seed_data = [
        ("Marcin", np.random.rand(16).astype(np.float32).tobytes()),
        ("rowerze", np.random.rand(16).astype(np.float32).tobytes()), # Correct spelling memory
        ("rower", np.random.rand(16).astype(np.float32).tobytes())
    ]
    cursor.executemany("INSERT INTO context_memories VALUES (?, ?)", seed_data)
    conn.commit()
    return conn

# ==========================================
# 2. ENCODER
# ==========================================
def text_to_vector(text, size=16):
    # Deterministic mapping based on chars
    vec = np.zeros(size, dtype=np.float32)
    for i, char in enumerate(text):
        idx = i % size
        vec[idx] += ord(char) / 255.0
    # Normalize
    norm = np.linalg.norm(vec)
    if norm > 0: vec = vec / norm
    return vec

def vector_to_text_approx(vec):
    # Just a placeholder for decoding visualization
    # In real usage, this would map back to nearest vocabulary word
    return f"Vec({vec[:2]}...)"

# ==========================================
# 3. LORENZ AGENT (Advanced)
# ==========================================
class LorenzAgent:
    def __init__(self, agent_id, ltm_conn):
        self.id = agent_id
        self.ltm_conn = ltm_conn

        # Lorenz State (Dynamical Core)
        self.lorenz_state = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        # Knowledge State (16D)
        self.vector_state = np.zeros(16, dtype=np.float32)

        # Parameters (Tuned for Stability)
        self.sigma = 5.0 # Was 10
        self.rho = 14.0 # Was 28
        self.beta = 8.0 / 3.0
        self.dt = 0.005 # Was 0.01 (slower time)

        self.stability_threshold = 50.0 # If Lorenz goes too far, it's drifting
        self.focus = 1.0
        self.drift_count = 0

    def step_lorenz(self):
        """Calculate next Lorenz step"""
        x, y, z = self.lorenz_state

        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z

        self.lorenz_state += np.array([dx, dy, dz]) * self.dt

        # Bounds Check (The "Cage")
        mag = np.linalg.norm(self.lorenz_state)
        if mag > self.stability_threshold:
            # Damping force to return to attractor center
            self.lorenz_state *= 0.95
            self.focus -= 0.05
        else:
            self.focus += 0.01

        self.focus = np.clip(self.focus, 0.0, 1.0)

    def process_input(self, input_vec):
        """
        Process 16D vector.
        Couples Lorenz State -> Vector State.
        """
        # 1. Update Dynamics
        self.step_lorenz()

        # 2. Map Lorenz (3D) to Vector (16D) perturbation
        # We repeat the 3D state to fill 16D space roughly
        perturbation = np.resize(self.lorenz_state, 16) * 0.01

        # 3. Apply Input Resonance
        # If input is strong, it overwrites drift
        self.vector_state = 0.8 * self.vector_state + 0.2 * input_vec + perturbation

        # 4. Check for Drift/Panic
        if self.focus < 0.2:
            self.rescue_from_ltm()

        return self.vector_state

    def rescue_from_ltm(self):
        """
        Agent is lost (Chaos too high). Queries LTM for stability.
        """
        self.drift_count += 1
        # Randomly pick a known concept to "ground" itself (Simulating context search)
        concepts = ["Marcin", "rower"]
        concept = random.choice(concepts)

        cursor = self.ltm_conn.cursor()
        cursor.execute("SELECT vector FROM context_memories WHERE keyword=?", (concept,))
        row = cursor.fetchone()

        if row:
            # Restore state from LTM
            self.vector_state = np.frombuffer(row[0], dtype=np.float32)
            self.lorenz_state = np.array([1.0, 1.0, 1.0]) # Reset dynamics
            self.focus = 0.8 # Boost focus
            # print(f"   [Agent {self.id}] 🆘 RESCUE ACTIVATED: Grounded to '{concept}'")

# ==========================================
# 4. SWARM CONTROLLER
# ==========================================
class SwarmSentenceController:
    def __init__(self, num_agents=35):
        self.conn = setup_ltm()
        self.agents = [LorenzAgent(i, self.conn) for i in range(num_agents)]

    def process_sentence(self, sentence):
        print(f"🌀 PROCESSING SENTENCE: '{sentence}'")
        tokens = sentence.split()

        # Layer Map for 35 Agents:
        # 0-11: Sensory (Input) - 12 agents
        # 12-23: Associative - 12 agents
        # 24-34: Output - 11 agents

        overall_history = []

        for t_idx, token in enumerate(tokens):
            print(f"\n🔹 Token {t_idx+1}/{len(tokens)}: '{token}'")
            vec = text_to_vector(token)

            # --- LAYER 1: SENSORY (12 agents) ---
            l1_output = np.zeros(16, dtype=np.float32)
            for i in range(0, 12):
                out = self.agents[i].process_input(vec)
                l1_output += out
            l1_output /= 12.0

            # --- LAYER 2: ASSOCIATIVE (12 agents) ---
            l2_output = np.zeros(16, dtype=np.float32)
            for i in range(12, 24):
                # Takes L1 output + some noise from neighbors
                out = self.agents[i].process_input(l1_output)
                l2_output += out
            l2_output /= 12.0

            # --- LAYER 3: OUTPUT (11 agents) ---
            l3_output = np.zeros(16, dtype=np.float32)
            for i in range(24, 35):
                out = self.agents[i].process_input(l2_output)
                l3_output += out
            l3_output /= 11.0

            # Stats
            total_drift = sum([a.drift_count for a in self.agents])
            avg_focus = np.mean([a.focus for a in self.agents])
            print(f"   Swarm Focus: {avg_focus:.2f} | Total Rescues: {total_drift}")

            overall_history.append(l3_output)

            # Simulate "Thinking time" for dynamics to settle
            for _ in range(5):
                for a in self.agents: a.step_lorenz()

        self.conn.close()
        return overall_history

if __name__ == "__main__":
    swarm = SwarmSentenceController(num_agents=35)
    swarm.process_sentence("Dzis byłem na roweże z Marcinem")
