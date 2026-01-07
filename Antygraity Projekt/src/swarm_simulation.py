
import numpy as np
import time
import random

# ==========================================
# 1. ENCODER / DECODER (Deterministic Base)
# ==========================================
class Codec:
    def __init__(self, vector_size=10):
        self.vector_size = vector_size

    def encode(self, text):
        """Simple ASCII normalization to [0,1] vector"""
        # Take first N chars, pad if needed
        data = [ord(c) for c in text[:self.vector_size]]
        while len(data) < self.vector_size:
            data.append(0)

        # Normalize to 0.0 - 1.0 (assuming ASCII max 255)
        vector = np.array(data, dtype=np.float64) / 255.0
        return vector

    def decode(self, vector):
        """Reconstruct text from vector"""
        # Denormalize
        data = (vector * 255.0).astype(int)
        chars = [chr(c) for c in data if 32 <= c <= 126] # Printable range
        return "".join(chars)

# ==========================================
# 2. SWARM AGENT (Chaotic Logic)
# ==========================================
class Agent:
    def __init__(self, agent_id, vector_size):
        self.id = agent_id
        self.state = np.random.rand(vector_size) # Internal Vector State
        self.vector_size = vector_size

        # DNA / Parameters
        self.chaos_r = 3.9 + (random.random() * 0.1) # 3.9 - 4.0 (Deep Chaos)
        self.mutation_rate = 0.01
        self.coupling_strength = 0.3
        self.focus = 0.0 # Focus level (0.0 - 1.0)

        # Metrics History
        self.history = []

    def stimulate(self, amount=0.2):
        """Boost focus"""
        self.focus += amount
        if self.focus > 1.0: self.focus = 1.0

    def decay_focus(self):
        """Natural loss of focus over time"""
        self.focus *= 0.95

    def chaos_map(self, x):
        """Logistic Map: f(x) = r * x * (1-x)"""
        return self.chaos_r * x * (1.0 - x)

    def process(self, input_signal=None):
        """
        Main processing step.
        1. Apply Chaos to internal state.
        2. Couple with Input Signal (Resonance).
        3. Apply Mutation (if active).
        """
        # A. Internal Dynamics (Chaos)
        # Focus determines how "stable" the chaos is.
        # High focus = more adherence to map. Low focus = random drift.
        if self.focus > 0.1:
            next_state = self.chaos_map(self.state)
        else:
            # Random drift if not focused ("sleepy")
            next_state = self.state + np.random.normal(0, 0.01, self.vector_size)

        # B. Resonance (Coupling)
        if input_signal is not None:
            # Diffuse input into state
            next_state = next_state + self.coupling_strength * (input_signal - next_state)
            self.stimulate(0.1) # Input creates focus

        # C. Mutation (Random bit flips in logic)
        if random.random() < self.mutation_rate:
            # Mutate 'r' slightly
            self.chaos_r += np.random.normal(0, 0.05)
            self.chaos_r = np.clip(self.chaos_r, 3.5, 4.0)

        # Normalize to keep sensible
        self.state = np.clip(next_state, 0.0, 1.0)

        self.decay_focus()

        # Log stats
        self.history.append({
            'mean_val': np.mean(self.state),
            'focus': self.focus,
            'r': self.chaos_r
        })

        return self.state

# ==========================================
# 3. SWARM NETWORK (Topology)
# ==========================================
class Swarm:
    def __init__(self, size=4, vector_len=10):
        self.agents = [Agent(i, vector_len) for i in range(size)]
        self.codec = Codec(vector_len)
        # Hidden Layer (Ground Truth weights) - never mutates
        self.hidden_weights = np.eye(vector_len)

    def run_cycle(self, text_input=None):
        print("\n" + "="*80)
        print("🌀 STARTING SWARM SIMULATION (30 CYCLES)")
        print("="*80)

        # 1. Encode Input
        if text_input:
            input_vector = self.codec.encode(text_input)
            print(f"📥 INPUT: '{text_input}' -> Vector Sample: {input_vector[:3]}...")
        else:
            input_vector = np.zeros(self.agents[0].vector_size)

        # 2. Simulation Loop
        gateway_signal = input_vector

        report_data = []

        for cycle in range(30):
            # Topology:
            # Agent 0 (Gateway) gets Input
            # Agent 1 gets Agent 0
            # Agent 2 gets Agent 1
            # Agent 3 gets Agent 2 & 0 (Loopback)

            # --- AGENT 0 ---
            sig_0 = self.agents[0].process(input_signal=gateway_signal if cycle == 0 else None)

            # --- AGENT 1 (Listens to 0) ---
            sig_1 = self.agents[1].process(input_signal=sig_0)

            # --- AGENT 2 (Listens to 1) ---
            sig_2 = self.agents[2].process(input_signal=sig_1)

            # --- AGENT 3 (Listens to 2 AND 0 - Resonance Loop) ---
            # Average signal from 2 and 0
            mixed_input = (sig_2 + sig_0) / 2.0
            sig_3 = self.agents[3].process(input_signal=mixed_input)

            # --- STIMULUS CHECK ---
            # If cycle > 15, inject "Boost" to Agent 2 (simulating distant event)
            if cycle == 15:
                print(f"   >>> ⚡ INJECTING STIMULUS (FOCUS BOOST) TO AGENT 2 <<<")
                self.agents[2].stimulate(1.0)

            # Log snapshot
            report_data.append([a.history[-1] for a in self.agents])

            # Print minimal status
            foc_strs = [f"A{i}:F={a.focus:.2f}" for i, a in enumerate(self.agents)]
            print(f"Cycle {cycle+1:02d} | {' | '.join(foc_strs)}")

        # 3. Final Decoding
        print("\n" + "="*80)
        print("📊 FINAL REPORT")
        print("="*80)

        # We take the state of Agent 3 (End of chain)
        final_vector = self.agents[3].state

        # Apply Hidden Layer (Identity in this MVP, but represents 'Ground Truth' filtering)
        filtered_vector = np.dot(final_vector, self.hidden_weights)

        decoded_text = self.codec.decode(filtered_vector)
        print(f"📤 OUTPUT (Agent 3 State Mapped): '{decoded_text}'")

        print("\n📈 CHAOS ANALYTICS:")
        for i, a in enumerate(self.agents):
            start_r = a.history[0]['r']
            end_r = a.history[-1]['r']
            print(f"Agent {i}: Chaos (r) {start_r:.4f} -> {end_r:.4f} | Mutated: {start_r != end_r}")
            print(f"         Final Focus: {a.focus:.4f}")

if __name__ == "__main__":
    swarm = Swarm(size=4, vector_len=12)
    swarm.run_cycle("CHAOS_AI")
