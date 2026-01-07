
import numpy as np
import random
import time
from swarm_acn import BioAgent, lorenz_energy, acn_tick
from swarm_simulation import Codec

# ==========================================
# 1. GENOME DEFINITION
# ==========================================
class Genome:
    def __init__(self):
        # SHAPE
        self.vector_capacity = 16 # Int, size of vector
        self.processing_speed = 0.9 # Float 0.0-1.0

        # BEHAVIOR
        self.bridge_capability = random.choice([True, False])
        self.fallback_strategy = "WAIT" # or "RETRY", "PANIC"

        # SPECIALIZATION
        self.focus_indices = random.sample(range(16), 3) # Indices to prioritize
        self.role = random.choice(["quantizer", "compressor", "cache"])

    def mutate(self):
        """
        Evolutionary logic (+1% rule from docs)
        """
        mutation_log = []

        # SHAPE: Grow capacity (simulated by int increment check)
        if random.random() < 0.1:
            self.vector_capacity = int(self.vector_capacity * 1.01) + 1
            mutation_log.append("SHAPE: Capacity Up")

        # SHAPE: Speed drift
        shift = np.random.normal(0, 0.01)
        self.processing_speed = np.clip(self.processing_speed + shift, 0.1, 1.0)

        # BEHAVIOR: Toggle Bridge
        if random.random() < 0.05:
            self.bridge_capability = not self.bridge_capability
            mutation_log.append(f"BEHAVIOR: Bridge -> {self.bridge_capability}")

        return mutation_log

# ==========================================
# 2. EVO-AGENT (BioAgent + Genome)
# ==========================================
class EvoAgent(BioAgent):
    def __init__(self, agent_id):
        super().__init__(agent_id, vector_size=16) # Start with base 16
        self.genome = Genome()

    def process_cycle(self):
        # 1. Mutate?
        logs = []
        if random.random() < 0.05: # 5% chance per cycle in this fast sim
            logs = self.genome.mutate()

        # 2. Simulate Processing Latency based on Speed
        # Speed 1.0 = 0 delay. Speed 0.1 = high delay.
        # We simulate this CPU cost artificially
        base_cost = 0.001 # 1ms
        real_cost = base_cost / self.genome.processing_speed
        time.sleep(real_cost)

        return logs

# ==========================================
# 3. SWARM EVOLUTION CONTROLLER
# ==========================================
class SwarmEvoController:
    def __init__(self, size=10):
        self.agents = [EvoAgent(i) for i in range(size)]
        self.codec = Codec(vector_size=16)

    def run_experiment(self, input_text, cycles=10):
        print(f"\n{'='*60}")
        print(f"🧬 STARTING EVOLUTIONARY CYCLE: '{input_text}'")
        print(f"{'='*60}")

        # 1. Input Injection (Gateway Agent 0)
        input_vec = self.codec.encode(input_text)
        self.agents[0].stm = input_vec.copy()
        self.agents[0].activity = np.ones(16) # Fully excited

        # Metrics
        total_interference = 0.0
        start_time = time.time()
        mutation_history = []

        for c in range(cycles):
            cycle_start = time.time()

            # A. ACN Stimulation (Interference Source)
            # We run N ticks of ACN per cycle to simulate noise
            cycle_energy = 0.0
            for _ in range(5):
                res = acn_tick(self.agents) # using imported function, need to adapt agent class?
                # acn_tick expects 'sender.activity', 'stm', 'l_state' which EvoAgent inherits from BioAgent. Great.
                # acn_tick returns tuple or None.
                if res:
                    sid, rid, nrg = res
                    cycle_energy += nrg

            total_interference += cycle_energy

            # B. Agent Processing & Evolution
            for a in self.agents:
                m_logs = a.process_cycle()
                if m_logs:
                    mutation_history.extend([f"Cycle {c} A{a.id}: {l}" for l in m_logs])

            # C. Output Check (Agent 9)
            # Reconstruct text from Agent 9
            curr_vec = self.agents[-1].stm
            decoded = self.codec.decode(curr_vec)

            # Log
            elapsed = (time.time() - start_time) * 1000
            print(f"Cy {c} | Int: {cycle_energy:.2f} | Latency: {elapsed:.0f}ms | Out: '{decoded}'")

        print(f"\n{'='*60}")
        print("📊 FINAL METRICS")
        print(f"Total Cycles: {cycles}")
        print(f"Total Interference (Chaos Energy): {total_interference:.4f}")
        print(f"Total Mutation Events: {len(mutation_history)}")
        print("Mutations:")
        for m in mutation_history[:5]: print(f" - {m}")
        if len(mutation_history) > 5: print(" ...")

if __name__ == "__main__":
    swarm = SwarmEvoController()
    swarm.run_experiment("EVOLUTION_TEST")
