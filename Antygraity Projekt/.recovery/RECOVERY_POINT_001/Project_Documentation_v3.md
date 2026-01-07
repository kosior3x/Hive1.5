# AI Mentor / Vis-Sol Project Documentation
**Master Design Document & Technical Specification**
**Version:** 3.0 (Evolutionary Bio-Swarm)
**Date:** December 15, 2025
**Status:** PROTOTYPE VALIDATED

---

## 1. Executive Summary
The AI Mentor system is a **Bio-Inspired Evolutionary Swarm** designed for offline, edge-based intelligence. Unlike LLMs, it does not predict tokens; it **processes semantic states** through a living colony of agents.
Key characteristics:
*   **Chaotic Dynamics**: Uses Lorenz Attractors for creativity and state transition.
*   **Peer-to-Peer Stimulation (ACN)**: Agents "nudge" each other dendritically.
*   **Evolutionary Genome**: Agents mutate parameters (+1% growth) over time.
*   **Context Stability**: Anchored by Long-Term Memory (LTM) attractors.

## 2. Core Architecture

### 2.1. The Agent (Bio-Neuron)
Each agent is an autonomous entity containing:
1.  **STM (Short Term Memory)**: A floating-point vector (e.g., 16D) representing current thought.
2.  **Origin (DNA)**: The base resting state (0.0 reference).
3.  **Lorenz Core**: Internal chaotic oscillator `(x, y, z)` generating scalar energy.
4.  **Genome**:
    *   `SHAPE`: Vector capacity, Speed.
    *   `BEHAVIOR`: Bridge capability, Fallback strategy.
    *   `SPECIALIZATION`: Focus indices.

### 2.2. The Network (Swarm)
*   **Topology**: Dynamic Mesh. Agents connect via random P2P dendrites.
*   **Layers**:
    *   **Sensory**: Input vectors (Text -> Codec -> Vector).
    *   **Associative**: Chaos-driven processing.
    *   **Output**: Consensus readout.

### 2.3. Communication (Bio-ACN)
Instead of message passing, agents use **ACN (Activator of Neural Stimuli)**:
*   **Logic**: `Target = Base - abs(Incoming - Base) * ChaosEnergy`
*   **Constraint**: Changes must be **regenerative** (median stability), never explosive.
*   **Effect**: A subtle "nudge" that propagates meaning without destroying context.

## 3. Evolutionary Mechanics

### 3.1. Mutation
Every N cycles, an agent's genome mutates:
*   **Capacity Growth**: `Capacity *= 1.01` (1% Rule).
*   **Behavior Flip**: Toggles bridge/routing flags.
*   **Speed Drift**: Processing latencies shift naturally.

### 3.2. Degradation (Rolling Decay)
*   Without stimulus, STM decays linearly back to Origin.
*   This mimics biological forgetting and prevents saturation.

## 4. Technical Stack
*   **Language**: Python 3.10+
*   **Math**: NumPy (Vector operations, Quantization), SciPy (Integrators).
*   **Persistence**: SQLite (LTM Context Anchors).
*   **Hardware Target**: Low-power Mobile (e.g., OPPO A78, Snapdragon 680).

## 5. Validated Components (POCs)
1.  **Chaos Sync**: Coupled Logistic Maps sync perfectly without data exchange.
2.  **Context Latching**: Swarm stabilizes around "Bike Context" given related keywords.
3.  **Bio-ACN**: P2P dendrites produce stable, non-hallucinating drift.
4.  **Evolution**: Mutations occur in real-time correlation with processing.

---

## 6. Future Roadmap (Phase 4)
*   **Android Porting**: Wrap Python core in Chaquopy/Kivy.
*   **Visualizer**: 3D Phase Space plot of Swarm Thought.
*   **Hardware Interface**: Sensor input (Mic/Cam) direct to Vector.
