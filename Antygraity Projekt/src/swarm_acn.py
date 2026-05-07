import numpy as np
import random
import time

# ==========================================
# 1. CHAOS ENERGY (SCALAR)
# ==========================================
def lorenz_energy(x, y, z, sigma=10, rho=28, beta=8/3, dt=0.01, steps=10):
    """
    Generates a scalar 'Energy' value from the Lorenz attractor.
    Instead of using the vector (x,y,z) directly, we use the magnitude/evolution
    to determine 'How strong is the stimulus?'.
    """
    for _ in range(steps):
        dx = sigma * (y - x) * dt
        dy = (x * (rho - z) - y) * dt
        dz = (x * y - beta * z) * dt
        x, y, z = x + dx, y + dy, z + dz

    # Energy = Magnitude of the chaos state (normalized roughly to 0.0-1.0 range usually)
    # We use abs(x) scaled down to be a reasonable multiplier
    energy = abs(x) / 20.0
    return energy

# ==========================================
# 2. BIO-AGENT (STM + Dendrites)
# ==========================================
class BioAgent:
    def __init__(self, agent_id, vector_size=5):
        self.id = agent_id
        # Origin: The "Resting State" or "Genetic Memory" (Point 0.0)
        self.origin = np.random.rand(vector_size).astype(np.float32)
        # STM: Short Term Memory (starts at Origin)
        self.stm = self.origin.copy()
        # Activity: How excited/stimulated the agent is (0.0 - 1.0)
        self.activity = np.zeros(vector_size, dtype=np.float32)

        # Internal Lorenz State for generating its own unique energy signature
        self.l_state = (random.random(), random.random(), random.random())

    def receive_stimulus(self, stimulus, energy):
        """
        Dendritic logic:
        1. Calculate Target = Descent towards stimulus intensity.
        2. Apply Median Filter = Ensure we don't overshoot Base or Current.
        3. Apply EPSP = If stuck at base, apply micro-nudge.
        """
        EPS = 0.02 # Micro-deviation threshold (EPSP)

        for i in range(len(stimulus)):
            base = self.origin[i]
            current = self.stm[i]
            incoming = stimulus[i] # The value from the sender

            # Target is a "Descent" or "Modification" driven by chaos energy
            # We assume the stimulus pulls us 'away' from origin by some delta
            # Logic: target = base - (distance_to_incoming * energy)
            # This ensures we generally 'relax' or 'modify' downwards/inwards, never exploding up.
            target = base - abs(incoming - base) * energy

            # CONSTRAINT 1: Micro-Nudge (EPSP)
            # If we are exactly at resting state, we need a kick to start moving
            if abs(current - base) < 1e-6:
                 # Sign depends on incoming direction vs base
                 direction = -1 if incoming < base else -1 # Always subtract based on 'Downwards' rule?
                 # Actually user said "never look above, only down"
                 self.stm[i] = base - EPS * energy

            # CONSTRAINT 2: Median Stability
            # strictly limit the new value to be between [Base, Target, Current]
            # This prevents wild jumps.
            elif target < current:
                self.stm[i] = np.median([current, target, base])

                # Boost activity if we successfully moved
                self.activity[i] = min(1.0, self.activity[i] + energy * 0.2)

    def decay(self, rate=0.01):
        """
        Biological forgetting.
        STM slowly returns to Origin if not stimulated.
        Activity drops.
        """
        # Linear decay towards zero/origin
        self.stm = self.stm * (1.0 - rate) + self.origin * rate
        self.activity *= 0.95

# ==========================================
# 3. ACN TICK (P2P Logic)
# ==========================================
def acn_tick(agents):
    # 1. Select Random Pair (Dendritic Connection)
    if len(agents) < 2: return
    sender, receiver = random.sample(agents, 2)

    # 2. Transmit Stimulus
    # We send the full STM vector. The receiver will handle processing.
    stimulus = sender.stm.copy()

    # 3. Generate Chaos Energy
    # sender uses its internal lorenz state
    lx, ly, lz = sender.l_state
    energy = lorenz_energy(lx, ly, lz)

    # Update sender's lorenz state
    sender.l_state = (lx + 0.01, ly + 0.01, lz + 0.01) # Simple tick, real lorenz inside func

    # 4. Transmit
    receiver.receive_stimulus(stimulus, energy)

    return sender.id, receiver.id, energy

# ==========================================
# 4. SIMULATION RUNNER
# ==========================================
if __name__ == "__main__":
    print(f"{'='*60}")
    print("🧬 BIO-INSPIRED ACN: P2P DENDRITIC STIMULATION")
    print(f"{'='*60}")

    # Init Agents
    agents = [BioAgent(i, vector_size=5) for i in range(5)]

    print("\n--- Initial States (Origin == STM) ---")
    for a in agents:
        print(f"Agent {a.id}: {a.stm[:3]}...")

    print("\n--- Running 50 ACN Ticks ---")

    for t in range(50):
        # Stimulation
        sid, rid, nrg = acn_tick(agents)

        # Degradation (occurs every tick for everyone)
        for a in agents:
            a.decay(rate=0.05)

        if t % 10 == 0:
            print(f"Tick {t:02d}: A{sid} -> A{rid} (E={nrg:.2f})")

    print("\n--- Final STM States (Check for Stable Drift) ---")
    changes_detected = False
    for a in agents:
        diff = np.linalg.norm(a.stm - a.origin)
        print(f"Agent {a.id}: {a.stm[:3]}... | Drift: {diff:.6f}")
        if diff > 1e-5: changes_detected = True

    if changes_detected:
        print("\n✅ SUCCESS: STM drifted stably from Origin (biological plasticity).")
    else:
        print("\n❌ WARNING: No changes detected. EPSP threshold might be too low.")
