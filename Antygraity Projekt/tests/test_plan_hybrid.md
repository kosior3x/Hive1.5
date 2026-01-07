# Hybrid System Test Suite Implementation Plan

## 1. Mock Semantic Engine (`utils/semantic_mock.py`)
Since we don't have a real 320D embedding model loaded (Word2Vec/BERT), we must mock it intelligently to test **logic** not **data**.
*   **Vector Space**: 320D vectors.
*   **Categories**: Pre-defined cluster centers (e.g., [1,0,...] for Animals, [0,1,...] for Vehicles) + noise.
*   **Logic**: `encode("kot")` -> returns `Animal_Center + random_noise`.

## 2. Test Classes (`tests/test_hybrid_full.py`)

### A. SemanticTest
*   **Single Word**: Encodes 20 words, processes, decodes. Checks if output category matches input.
*   **Context**: Feeds "Bike" related tokens. Checks if distance to "Bike_Center" decreases.

### B. PerformanceTest
*   **Agent Scaling**: Loop 1 to 100 agents. Measure time/memory.
*   **Token Scaling**: Loop 1 to 1000 tokens.

### C. ChaosTest
*   **Sensitivity**: Run with Sigma=3, 5, 10, 15. Check output variance.
*   **Attention**: Capture CSA masks. Check if "Cat" activates similar neurons to "Dog".

### D. ComparativeTest
*   **Baseline**: Run old `BioAgent` (no CSA).
*   **Hybrid**: Run new `HybridAgent`.
*   **Compare**: Distance to target, Processing time.

## 3. Execution & Reporting
*   Script will append results to a big Dictionary.
*   Final step: Dump to `Hybrid_Test_Results.json`.
*   Optional: Print markdown summary to console.
