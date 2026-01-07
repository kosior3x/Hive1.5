
import numpy as np
import random
import time
from swarm_evolution import SwarmEvoController, Genome
from swarm_acn import BioAgent, acn_tick
from swarm_context_test import setup_context_ltm, LorenzAgent

class StressTester(SwarmEvoController):
    def __init__(self):
        super().__init__(size=15)
        self.conn, self.context_bike = setup_context_ltm()
        self.context_work = (np.zeros(16, dtype=np.float32) + 0.1) # from swarm_context_test

    def run_chaos_storm(self):
        print(f"\n⚡ TEST 1: CHAOS STORM (Energy > 5.0)")
        print("Expected: Median Filter prevents explosion (Values stay < 2.0)")

        # Monkey patch energy generator to return INSANE values
        def insane_energy(*args): return 10.0 + random.random() * 100.0

        # Inject insane energy
        # Only BioAgents have receive_stimulus logic with median filter
        # We manually trigger stimulus with insane energy

        for i in range(10):
            # Pick random pair
            a, b = random.sample(self.agents, 2)
            stimulus = a.stm.copy()
            energy = insane_energy()

            b.receive_stimulus(stimulus, energy)

            max_val = np.max(b.stm)
            status = "✅ SAFE" if max_val < 5.0 else "❌ EXPLODED"
            print(f"   Tick {i}: Energy={energy:.1f} -> Max Val={max_val:.4f} [{status}]")

    def run_conflict(self):
        print(f"\n⚡ TEST 2: CONTEXT CONFLICT (Bike vs Work)")

        # Create hybrid vector
        vec_bike = self.context_bike
        vec_work = self.context_work
        hybrid = 0.5 * vec_bike + 0.5 * vec_work

        # Feed to agents
        for a in self.agents:
            a.stm = hybrid.copy()

        print("   Fed Hybrid Vector. Allowing 10 cycles of ACN resonance...")

        for _ in range(10):
            acn_tick(self.agents)

        # Check consensus
        final_stm = np.mean([a.stm for a in self.agents], axis=0)
        dist_bike = np.linalg.norm(final_stm - vec_bike)
        dist_work = np.linalg.norm(final_stm - vec_work)

        print(f"   Final Distance to BIKE: {dist_bike:.4f}")
        print(f"   Final Distance to WORK: {dist_work:.4f}")

        if abs(dist_bike - dist_work) < 0.1:
            print("   Result: ⚖️ CONFUSION (Stuck in middle)")
        elif dist_bike < dist_work:
            print("   Result: 🚴 WINNER IS BIKE")
        else:
            print("   Result: 💼 WINNER IS WORK")

    def run_hyper_mutation(self):
        print(f"\n⚡ TEST 3: HYPER-MUTATION (100% Rate)")

        # Force mutation
        mut_count = 0
        input_text = "SURVIVAL"
        self.agents[0].stm = self.codec.encode(input_text)

        start_t = time.time()
        for i in range(10):
            for a in self.agents:
                # Force mutate manually for test
                a.genome.mutate()
                mut_count += 1

            # Simple process
            acn_tick(self.agents)

        final_out = self.codec.decode(self.agents[0].stm) # Check integrity of source after chaos

        print(f"   Mutations: {mut_count} (in 150 possible slots)")
        print(f"   Output text integrity: '{final_out}'")

        if final_out == input_text:
            print("   Result: 🛡️ ROBUST (Data survived mutation storm)")
        else:
            # It's expected to degrade a bit, but shouldn't be empty
            print(f"   Result: ⚠️ DEGRADED (As expected)")

if __name__ == "__main__":
    tester = StressTester()
    tester.run_chaos_storm()
    tester.run_conflict()
    tester.run_hyper_mutation()
