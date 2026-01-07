# Bio-Swarm + CSA Hybrid Architecture

**Version:** 1.0
**Date:** December 15, 2025

## 1. Core Philosophy
The Hybrid System combines two previously antagonistic approaches:
*   **Bio-Swarm (Chaos):** Provides adaptability, exploration, and resilience (The "Life").
*   **CSA (Structure):** Provides semantic precision, selective attention, and embedding processing (The "Logic").

## 2. Key Architectural Breakthroughs

### 2.1. Semantic Diffusion (The "Flow")
*   **Problem:** Traditional ACN used semi-homeostatic decay (`base - abs(diff)`), which forced vectors towards zero/origin, erasing the sign (direction) of semantic embeddings.
*   **Solution:** **Semantic Diffusion**.
    ```python
    target = base + (incoming - base) * energy_coefficient
    ```
*   **Impact:** This allows semantic vectors to "flow" through the swarm network. Chaos energy modulates the *rate* of diffusion, not the direction. High energy = fast information transfer; Low energy = local memory retention.

### 2.2. Simulated Pre-Training (Identity Initialization)
*   **Problem:** Initializing CSA weights with pure Gaussian noise effectively scrambled the meaningful embedding signals coming from the input, leading to massive drift (20% accuracy).
*   **Solution:** **Identity + Noise**.
    ```python
    self.W = np.eye(dim) + np.random.randn(dim, dim) * 0.05
    ```
*   **Impact:** Acts as a "Skip Connection" by default. It simulates a network that has already learned to pass features through, allowing us to validate the swarm dynamics without needing a computationally expensive pre-training phase.

### 2.3. LTM Gyroscope (The "Anchor")
*   **Problem:** Without ground truth correction, chaotic interactions eventually cause vector drift into arbitrary spaces (Attractor Collapse).
*   **Solution:** **Periodic LTM Anchoring**.
    ```python
    if cycle % 2 == 0:
        center = ltm.get_nearest_concept(agent.stm)
        agent.stm = (1 - pull) * agent.stm + pull * center
    ```
*   **Impact:** Acts like a gyroscope or proprioception. It provides a gentle corrective force that keeps the agent's internal state grounded in semantic reality while still allowing for local exploration.

## 3. System Scalability
*   **Complexity:** **O(N)** (Linear).
*   **Overhead:** Fixed cost per agent (~2.6ms).
*   **Sparsity:** ~37.5% active neurons via Lorenz Masking.
*   **Target:** Mobile/Edge Deployment.

## 4. Signal Flow
1.  **Input:** Text -> FastText/BERT -> Vector (320D).
2.  **Injection:** Vector injected into Entry Agents.
3.  **Process (Cycle):**
    *   **CSA:** Mask stimulus -> Identity Projection -> Self-Attention.
    *   **ACN:** Semantic Diffusion to neighbors (P2P).
    *   **LTM:** Anchor check (every N cycles).
4.  **Output:** Consensus of Output Agents -> Decoder.
