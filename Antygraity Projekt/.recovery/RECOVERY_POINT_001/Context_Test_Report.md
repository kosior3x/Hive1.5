# Context Hypothesis Test Report

## Experiment Configuration
*   **Swarm Size**: 15 Agents (3 Layers: Sensory, Associative, Output)
*   **Dynamics**: Lorenz Attractor (Tuned: Sigma=5.0, Rho=14.0)
*   **Context Target**: "BIKE_CONTEXT" (Hidden vector in LTM)
*   **Input Sequence**: 4 Sentences telling a story about a bike trip.

## Results

### Distance Metric (Lower is Better)
The distance represents the deviation of the Swarm's "Collective Thought" from the hidden "Bike Context".

| Input Token | Distance (Run 1 - Chaos) | Distance (Run 2 - Tuned) | Analysis |
| :--- | :--- | :--- | :--- |
| **Start** | 0.00 | 1.15 | Initial state |
| '30' | 1.77 | 1.45 | Numeric data adds noise |
| 'Trasa' | 3.53 | **1.23** | Context word stabilizes swarm |
| 'nasmarować' | 7.21 | **1.34** | High stability maintained |
| **Final** | **7.49** | **1.70** | **Significant improvement** |

### Key Findings
1.  **Chaos Must Be Tamed**: In Run 1 (Standard Lorenz), the swarm exploded logarithmically away from the context (`Distance > 7.0`). The "Butterfly Effect" destroyed the meaning.
2.  **Damping Works**: By reducing `Sigma` (Velocity) and adding a `0.95` damping factor between layers, the swarm became "Sticky". It held onto the context (`Distance ~1.2`) even when processing neutral words like "była" or "super".
3.  **Resonance confirmed**: Explicit keywords ("łańcuch") acted as anchors, pulling the drifting swarm back towards the target context vector.

## Conclusion
The **Context Hypothesis is CONFIRMED**.
A chaotic swarm *can* maintain a stable semantic context over multiple sentences if:
1.  The chaos parameters are tuned to be "Edge of Chaos" (not deep chaos).
2.  There are frequent "Anchors" (keywords) triggering LTM Resonance.
3.  The network layout allows for feedback/damping.

## Recommendations for v2.0
*   Implement **Dynamic Sigma Control**: Increase chaos when input is novel (to learn), decrease when input matches LTM (to focus).
*   Add **Sleep Cycles**: Allow the swarm to settle (Distance -> 0) between sentences.
