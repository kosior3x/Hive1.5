
import numpy as np
import random
import time
from swarm_acn import BioAgent, lorenz_energy, acn_tick
# Note: Codec will be redefined inside this file or imported if robust enough in future

# ==========================================
# 0. CSA LAYER (Chaotic Sparse Attention)
# ==========================================
class CSALayer:
    """
    Implements:
    1. Lorenz-based Chaotic Masking (Sparse Attention).
    2. Dense Layers (Simulated simplified MLP).
    3. Skip Connections (0.7 Identity + 0.3 Transform).
    """
    def __init__(self, dim=16, sparsity=0.4):
        self.dim = dim
        self.sparsity = sparsity

        # Lorenz Core for Mask Generation
        self.lorenz_state = np.random.rand(3)

        # Weights (Simulated "Trained" State)
        # Instead of pure random noise (which destroys semantics),
        # we start with Identity matrix + slight perturbation.
        # This simulates that the network has learned to preserve features.
        self.W = np.eye(dim) + np.random.randn(dim, dim) * 0.05

    def step_lorenz(self, sigma=10.0):
        x, y, z = self.lorenz_state
        dt = 0.01
        # sigma is now dynamic
        rho, beta = 28.0, 8.0/3.0

        dx = sigma * (y - x) * dt
        dy = (x * (rho - z) - y) * dt
        dz = (x * y - beta * z) * dt

        self.lorenz_state += np.array([dx, dy, dz])
        return self.lorenz_state

    def forward(self, input_vec, sigma=10.0):
        # 1. Update Lorenz with dynamic sigma
        l_state = self.step_lorenz(sigma=sigma)

        # 2. Generate Binary Mask from Chaos
        # Project 3D chaos to Dim space interaction
        # We use a simple hash-like method to map 3D -> Dim
        chaos_projection = np.zeros(self.dim)
        for i in range(self.dim):
            # Complex mixing of x,y,z based on index i
            val = l_state[0] * np.sin(i) + l_state[1] * np.cos(i) + l_state[2]
            chaos_projection[i] = val

        # Percentile threshold for sparsity
        thresh = np.percentile(chaos_projection, self.sparsity * 100)
        mask = (chaos_projection > thresh).astype(float)

        # 3. Apply Sparse Processing (Attention)
        # Only process masked elements through W, others are 0 in transform
        transformed = np.dot(self.W, input_vec * mask)

        # 4. Skip Connection (Identity Preservation)
        # 70% Original, 30% Chaos Transform
        output = 0.7 * input_vec + 0.3 * transformed

        return {
            'output': output,
            'mask': mask,
            'lorenz_state': l_state
        }

# ==========================================
# 1. GENOME (Updated for 30 agents)
# ==========================================
class Genome:
    def __init__(self):
        # SHAPE
        self.vector_capacity = 16
        self.processing_speed = 0.9

        # BEHAVIOR
        self.bridge_capability = random.choice([True, False])
        self.fallback_strategy = "WAIT"

        # SPECIALIZATION
        self.focus_indices = random.sample(range(16), 3)
        self.role = random.choice(["quantizer", "compressor", "cache"])

    def mutate(self):
        mutation_log = []
        if random.random() < 0.1:
            self.vector_capacity = int(self.vector_capacity * 1.01) + 1
            mutation_log.append("SHAPE: Capacity Up")

        shift = np.random.normal(0, 0.01)
        self.processing_speed = np.clip(self.processing_speed + shift, 0.1, 1.0)

        if random.random() < 0.05:
            self.bridge_capability = not self.bridge_capability
            mutation_log.append(f"BEHAVIOR: Bridge -> {self.bridge_capability}")

        return mutation_log

# ==========================================
# 2. HYBRID AGENT (Evo + CSA)
# ==========================================
class HybridAgent(BioAgent):
    def __init__(self, agent_id):
        # Initialize BioAgent (STM, Origin, Lorenz Energy)
        super().__init__(agent_id, vector_size=16)

        self.genome = Genome()

        # NEW: CSA Layer integration
        self.csa = CSALayer(dim=16, sparsity=0.6) # 60% Sparse (40% Active)
        self.current_sigma = 10.0 # Default Chaos

    def adapt_to_context(self, context_type="learning"):
        """
        Faza 3: Adaptive Chaos Parameters
        Synchronized chaos - swarm + CSA work together.
        """
        if context_type == "learning":
            # Higher chaos for exploration
            self.current_sigma = 15.0
        elif context_type == "retrieval":
            # Lower chaos for stability
            self.current_sigma = 3.0
        else:
            self.current_sigma = 5.0

    def receive_stimulus(self, stimulus, energy):
        """
        Overriding BioAgent.receive_stimulus to implement Hybrid CSA Logic.
        """
        # A. Apply CSA with Adaptive Sigma
        csa_res = self.csa.forward(stimulus, sigma=self.current_sigma)
        masked_stimulus = csa_res['output']

        EPS = 0.02

        for i in range(len(self.stm)):
            base = self.origin[i]
            current = self.stm[i]
            incoming = masked_stimulus[i]

            # B. ACN Nudge Logic (Target Calculation)
            # UPDATED FOR SEMANTICS: Use linear diffusion instead of absolute decay
            # Old: target = base - abs(incoming - base) * energy (Destroys semantics)
            # New: target = base + (incoming - base) * energy (Preserves semantics)
            diff = incoming - base
            target = base + diff * energy

            # C. Median Stability
            if abs(current - base) < 1e-6:
                self.stm[i] = base - EPS * energy
            elif target < current:
                self.stm[i] = np.median([current, target, base])
                if csa_res['mask'][i] > 0:
                    self.activity[i] = min(1.0, self.activity[i] + energy * 0.2)

    def process_cycle(self):
        # 1. Decay (Return to Origin/Base State)
        # Instead of forgetting to zero, we drift back to our semantic origin
        decay_rate = 0.05 # Adjust speed of forgetting
        self.stm = self.stm * (1 - decay_rate) + self.origin * decay_rate

        # 2. Mutate (Evolve Chaos Parameters)
        logs = []
        if random.random() < 0.1: # 10% chance
            logs = self.genome.mutate()
            # Evolve sigma based on genome stability
            noise = np.random.normal(0, 0.5)
            self.current_sigma = max(1.0, min(20.0, self.current_sigma + noise))

        # 3. SELF-ATTENTION Processing (Internal thought)
        # Use Adaptive Sigma
        csa_res = self.csa.forward(self.stm, sigma=self.current_sigma)

        # Apply Median Stability
        safe_output = np.zeros_like(self.stm)
        for i in range(len(self.stm)):
            curr = self.stm[i]
            target = csa_res['output'][i]
            base = self.origin[i] # Anchor

            # Median Filter (Current state, New Thought, Base Instinct)
            safe_output[i] = np.median([curr, target, base])

        self.stm = safe_output


        # 3. Simulate Latency
        base_cost = 0.001
        real_cost = base_cost / self.genome.processing_speed
        time.sleep(real_cost)

        return logs

# Codec Placeholder (Mocking Semantic Embedding)
class SemanticCodec:
    def __init__(self, dim=16):
        self.dim = dim

    def encode(self, text):
        # Mock Semantic Embedding: Hash string to deterministic vector
        # In real CSA, this would be Word2Vec/BERT step
        np.random.seed(sum([ord(c) for c in text]))
        return np.random.rand(self.dim)

    def decode(self, vec):
        # Return summary
        return f"SemVec(Mean={np.mean(vec):.2f})"

# ==========================================
# 3. HYBRID SWARM CONTROLLER
# ==========================================
class HybridSwarmController:
    def __init__(self, size=30): # UPDATED to 30 as requested
        self.agents = [HybridAgent(i) for i in range(size)]
        self.codec = SemanticCodec(dim=16)

    def run_experiment(self, input_text, cycles=10):
        print(f"\n{'='*60}")
        print(f"🧬 HYBRID SWARM (CSA+Bio) | Agents: {len(self.agents)} | Input: '{input_text}'")
        print(f"{'='*60}")

        # 1. Input Injection (Semantic) - Broadcast to Swarm
        input_vec = self.codec.encode(input_text)
        for a in self.agents:
            a.stm = input_vec.copy()
            a.origin = input_vec.copy()

        total_interference = 0.0
        start_time = time.time()

        for c in range(cycles):
            # A. ACN Stimulation (P2P Nudge)
            # Acn_tick needs valid agents. HybridAgent inherits BioAgent, so it works.
            # But wait, acn_tick uses 'receive_stimulus'.
            # We should probably override receive_stimulus in HybridAgent if we want CSA there too?
            # For now, let's assume ACN purely stimulates STM, and 'process_cycle' (Step B) applies CSA organization.

            cycle_energy = 0.0
            for _ in range(10): # More ticks for larger swarm
                res = acn_tick(self.agents)
                if res:
                    cycle_energy += res[2]

            total_interference += cycle_energy

            # B. Processing (CSA + Evolution)
            for a in self.agents:
                a.process_cycle() # Applies CSA self-attention

            # C. Output Check (Agent 29)
            decoded = self.codec.decode(self.agents[-1].stm)
            elapsed = (time.time() - start_time) * 1000
            print(f"Cy {c} | Int: {cycle_energy:.2f} | Latency: {elapsed:.0f}ms | {decoded}")

if __name__ == "__main__":
    swarm = HybridSwarmController(size=30)
    swarm.run_experiment("CHAOS_AI_HYBRID")
