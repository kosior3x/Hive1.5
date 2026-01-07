
import sys
import os

# Add src and utils to path to ensure modules are found
sys.path.append(os.path.abspath('src'))
sys.path.append(os.path.abspath('utils'))

from swarm_brain import SwarmBrain
from semantic_mock import SemanticEngineMock

# Initialize
brain = SwarmBrain()
semantic = SemanticEngineMock()

# Test sequence
messages = [
    "Cześć! Jestem Kamil.",
    "Mam 28 lat i lubię programowanie w Pythonie.",
    "Pracuję nad projektem AI z chaosem.",
    "Mam rower i lubię jeździć.",
]

print("="*60)
print("VECTOR DB TEST")
print("="*60)

for i, msg in enumerate(messages, 1):
    print(f"\n[Message {i}] {msg}")
    brain.update_memory(msg, semantic)

    print(f"Vector DB size: {len(brain.vector_db)}")
    if brain.vector_db:
        items = list(brain.vector_db.items())[:5]
        for word, data in items:
            print(f"  - {word}: count={data['count']}")

print("\n" + "="*60)
print("FINAL STATE")
print("="*60)
print(f"Total concepts: {len(brain.vector_db)}")

if len(brain.vector_db) < 5:
    print("\n🚨 ERROR: Too few concepts learned!")
    print("Expected: 10+, Got:", len(brain.vector_db))
else:
    print("\n✅ SUCCESS: Vector DB is working!")
