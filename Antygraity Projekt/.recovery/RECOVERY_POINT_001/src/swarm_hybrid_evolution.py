
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

        # RojSTM Component: Positive Initialization (A1/STM Refinement)
        # Instead of 0.0, we start with a healthy baseline "vibrancy"
        self.activity = np.full(16, 0.3, dtype=np.float32)

        self.genome = Genome()
        self.hits = 0
        self.specialization_node = None # LTM concept this agent "cares" about

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
                    self.hits += 1 # Utility record

    def process_cycle(self, swarm_density=1.0):
        # Increment age
        self.age += 1

        # 1. Decay (Return to Origin/Base State)
        # Uses individual decay_rate (A1)
        # "decay != global", "dependent on age/history" - already randomized in __init__
        self.stm = self.stm * (1 - self.decay_rate) + self.origin * self.decay_rate

        # 2. Mutate (Evolve Chaos Parameters)
        # A3. Scalable Mutation Rate
        # Base rate 0.1, scales down as density increases (logarithmic damping)
        # If density is high (many agents), mutation should be lower to prevent chaos?
        # Checklist: "Mutation rate dependent on agent count... log/exp function"
        # Checklist: "Stagnation at 100+" -> Actually implies we need helper mutation if large?
        # "Staly mutation rate dziala dla 30, ale przy 100 stagnacja" -> We need HIGHER or SMARTER mutation for large swarms?
        # "Utrzymac rownowage".
        # "Low efficiency -> higher mutation".

        mutation_prob = 0.1 * (1.0 / (swarm_density ** 0.5))

        # C1. Age Pressure: Old agents mutate less, young agents more
        if self.age < 50:
            mutation_prob *= 1.5 # Young exploration (exploracja młodych)
        elif self.age > 200:
            mutation_prob *= 0.5 # Old stability (stabilność starszych)

        logs = []
        if random.random() < mutation_prob:
            logs = self.genome.mutate()
            # Evolve sigma based on genome stability
            noise = np.random.normal(0, 0.5)
            self.current_sigma = max(1.0, min(25.0, self.current_sigma + noise))

        # C1. Sigma decay with age (Zmniejszaj sigmę wraz z wiekiem)
        # Target sigma 5.0 for old agents, starting from high (e.g. 20)
        sigma_decay = 0.99
        self.current_sigma = self.current_sigma * sigma_decay + 5.0 * (1 - sigma_decay)

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
        # Professional Semantic Embedding (A2): Unit Vectors (Normalization)
        import hashlib
        seed = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16) % (2**32)
        np.random.seed(seed)

        # 1. Generate normal distribution
        vec = np.random.normal(0, 1.0, self.dim)

        # 2. Normalize to Unit Vector (L2 Norm)
        norm = np.linalg.norm(vec) + 1e-9
        return vec / norm

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

        # C2. Monitoring stats
        self.initial_genome_mean = self._avg_genome()
        self.saturation_time = None
        self.creation_time = time.time()

    def link_agents_to_ltm(self, vector_db):
        """
        KROK 2: Oblicz podobieństwo i wybierz najlepsze dopasowania.
        Dla każdego agenta wybiera pojęcie LTM o najwyższym Similarity >= Threshold (0.3).
        """
        if not vector_db: return
        threshold = 0.3
        concepts = list(vector_db.keys())

        linked_count = 0
        for a in self.agents:
            best_node = None
            best_sim = -1.0

            # Ensure agent origin is normalized
            v_agent = a.origin / (np.linalg.norm(a.origin) + 1e-9)

            for node in concepts:
                v_ltm = vector_db[node]['vector'] # Already normalized in Brain

                # Cosine Similarity (dot product of unit vectors)
                sim = np.dot(v_agent, v_ltm)

                if sim > best_sim and sim >= threshold:
                    best_sim = sim
                    best_node = node

            a.specialization_node = best_node
            if best_node: linked_count += 1

        if linked_count > 0:
            print(f"   🔗 [Linker] Semantically linked {linked_count}/{len(self.agents)} agents to LTM concepts.")

    def semantic_priming(self, input_vec, vector_db):
        """
        KROK 3: Aktualizacja STM z wykorzystaniem powiązań.
        Wzmacnia agentów, których przypisane pojęcia pojawiają się w bodźcach (resonance).
        """
        if not vector_db: return

        # Ensure input is unit vector
        input_vec = input_vec / (np.linalg.norm(input_vec) + 1e-9)

        # 1. Oblicz similarity bodźca (input_vec) z pojęciami LTM
        resonance_scores = {}
        for word, data in vector_db.items():
            v_ltm = data['vector']
            sim = np.dot(input_vec, v_ltm)
            # Threshold for global resonance
            if sim > 0.4:
                resonance_scores[word] = sim

        # 2. Boost aktywności powiązanych agentów
        boost_count = 0
        boost_val = 0.3 # Mocny boost semantyczny

        for a in self.agents:
            if a.specialization_node and a.specialization_node in resonance_scores:
                score = resonance_scores[a.specialization_node]
                # Dynamic boost based on resonance score
                a.activity = np.minimum(a.activity + (boost_val * score), 1.0)
                # Synergia z chaosem (eksploracja aktywnego tematu)
                a.current_sigma = min(30.0, a.current_sigma + 5.0 * score)
                boost_count += 1

        if boost_count > 0:
            print(f"   ⚡ [Priming] Semantic Justified Activation: {boost_count} agents resonating.")

    def log_stm_activity(self, cycle):
        """Log average STM activity per agent to CSV"""
        import csv
        import os
        filename = "swarm_stm_activity.csv"
        file_exists = os.path.isfile(filename)

        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                # Header: Cycle, A00_Activity, A01_Activity...
                header = ['cycle'] + [f'A{a.id:02d}' for a in self.agents]
                writer.writerow(header)

            row = [cycle] + [f"{np.mean(a.stm):.4f}" for a in self.agents]
            writer.writerow(row)

    def _avg_genome(self):
        """Average genome speed/capacity for drift tracking"""
        if not self.agents: return 0
        return np.mean([a.genome.processing_speed for a in self.agents])

    def run_experiment(self, input_text, cycles=10, brain=None):
        print(f"\n{'='*60}")
        print(f"🧬 HYBRID SWARM (CSA+Bio) | Agents: {len(self.agents)} | Input: '{input_text}'")
        print(f"{'='*60}")

        # 1. Input Injection (Semantic) - Broadcast to Swarm WITH NOISE (A1)
        input_vec = self.codec.encode(input_text)

        # RojSTM Link Check: If new concepts learned, update links
        if brain and brain.vector_db and (not self.agents[0].specialization_node or random.random() < 0.1):
            self.link_agents_to_ltm(brain.vector_db)

        for a in self.agents:
            # A1. Desynchronization: Higher Micro-noise per agent (increased to 0.1)
            noise = np.random.normal(0, 0.1, size=self.codec.dim)
            a.stm = input_vec + noise
            # Origin also biased to prevent perfect attractor
            a.origin = input_vec + (noise * 0.3)

        total_interference = 0.0
        start_time = time.time()

        for c in range(cycles):
            # RojSTM: Dynamic Priming
            if brain and brain.vector_db:
                self.semantic_priming(input_vec, brain.vector_db)

            # Log Activity (CSV Optimization)
            self.log_stm_activity(c)

            # A. ACN Stimulation (P2P Nudge)
            cycle_energy = 0.0
            for _ in range(max(10, len(self.agents)//3)): # Adapt ticks to population
                res = acn_tick(self.agents)
                if res:
                    cycle_energy += res[2]

            total_interference += cycle_energy

            # B. Processing (CSA + Evolution)
            # A3. Density factor
            density = len(self.agents) / 30.0
            for a in self.agents:
                a.process_cycle(swarm_density=density)

            # A4. Dynamic Growth Logic (Check every 5 cycles)
            # Conditions: High saturation or Low variance (stagnation)
            if c % 5 == 0 and len(self.agents) < 100:
                # Calculate variance (Diversity of thought across agents)
                stms = np.array([a.stm for a in self.agents])
                # Variances per dimension, then averaged
                global_var = np.mean(np.var(stms, axis=0))

                # If variance is too low (collapsed) or energy very high (need more capacity)
                if global_var < 0.001 or cycle_energy > len(self.agents) * 0.8: # Threshold adjusted for axis variance
                    new_count = min(5, 100 - len(self.agents))
                    if new_count > 0:
                        start_id = self.agents[-1].id + 1
                        print(f"   🌱 [GROWTH] Spawning {new_count} new agents (Var={global_var:.4f})")
                        for i in range(new_count):
                            new_agent = HybridAgent(start_id + i)
                            # Fresh genome, high sigma for exploration
                            new_agent.current_sigma = 20.0
                            new_agent.process_cycle() # Init stats
                            self.agents.append(new_agent)

            decoded = self.codec.decode(self.agents[-1].stm)
            elapsed = (time.time() - start_time) * 1000

            # BLOCK 5. Monitoring (C2)
            stms = np.array([a.stm for a in self.agents])
            var = np.mean(np.var(stms, axis=0))

            # BLOCK 4. Selection & Removal (C1 Extended)
            # Threshold: older than 300 cycles OR very low activity/utility
            if c % 10 == 0 and len(self.agents) > 50:
                # Remove 1 least 'useful' (here: random old one for demo, real utility would track hits)
                old_agents = [a for a in self.agents if a.age > 300]
                if old_agents:
                    victim = random.choice(old_agents)
                    print(f"   💀 [DEATH] Agent {victim.id} removed (Age={victim.age})")
                    self.agents.remove(victim)

            print(f"Cy {c} | Agents: {len(self.agents)} | Var: {var:.4f} | Int: {cycle_energy:.2f} | {decoded}")

            # BLOCK 5. Regression Control
            if var < 0.0005:
                print(f"   ⚠️ [ALERT] Semantic Collapse Detected. Boosting Sigma.")
                for a in self.agents:
                    a.current_sigma = min(25.0, a.current_sigma + 5.0)

    def monitor_regression(self, brain=None):
        """Block 5: Automatic health check (C2)"""
        if not self.agents: return {}

        stms = np.array([a.stm for a in self.agents])
        var = np.mean(np.var(stms, axis=0))

        # C2. Genome Drift
        current_g = self._avg_genome()
        drift = abs(current_g - self.initial_genome_mean)

        # C2. Unrelated Similarity (A2)
        # If brain is provided, check random words in Vector DB
        unrelated_sim = 0
        if brain and len(brain.vector_db) > 10:
            words = list(brain.vector_db.keys())
            sims = []
            for _ in range(20):
                w1, w2 = random.sample(words, 2)
                # Skip if too similar by string (e.g. variants)
                if w1[:3] == w2[:3]: continue

                v1, v2 = brain.vector_db[w1]['vector'], brain.vector_db[w2]['vector']
                s = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
                sims.append((s + 1) / 2)
            unrelated_sim = np.mean(sims) if sims else 0
        else:
            # Fallback to internal agent similarity
            sims = []
            if len(self.agents) > 2:
                for _ in range(10):
                    a1, a2 = random.sample(self.agents, 2)
                    s = np.dot(a1.stm, a2.stm) / (np.linalg.norm(a1.stm) * np.linalg.norm(a2.stm) + 1e-9)
                    sims.append(s)
            unrelated_sim = np.mean(sims) if sims else 0

        return {
            'variance': var,
            'population': len(self.agents),
            'avg_age': np.mean([a.age for a in self.agents]),
            'genome_drift': drift,
            'unrelated_sim': unrelated_sim
        }

if __name__ == "__main__":
    swarm = HybridSwarmController(size=30)
    swarm.run_experiment("CHAOS_AI_HYBRID")
