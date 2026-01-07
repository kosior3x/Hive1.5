
import sys
import os
import time
import numpy as np

# Setup paths
sys.path.append(os.path.abspath('src'))
sys.path.append(os.path.abspath('utils'))

from swarm_brain import SwarmBrain
from swarm_hybrid_evolution import HybridSwarmController
from semantic_mock import SemanticEngineMock

def simulate_gui_flow():
    print("🚀 INITIALIZING SYSTEM SIMULATION...")

    # Clean up previous memory for a fresh test of the new feature
    if os.path.exists("swarm_ltm_store.pkl"):
        try:
            os.remove("swarm_ltm_store.pkl")
            print("   🧹 Removed old memory file for fresh test.")
        except:
            pass

    # 1. Initialize Components
    try:
        brain = SwarmBrain()
        swarm = HybridSwarmController(size=30)
        # Use SemanticMock as standard for now
        swarm.codec = SemanticEngineMock()
        semantic_engine = swarm.codec
        print("   ✅ Components initialized successfully.")
    except Exception as e:
        print(f"   ❌ FATAL ERROR during init: {e}")
        return

    # Define conversation flow
    conversation = [
        "Cześć! Chcę Cię czegoś nauczyć.",
        "Kryptowaluta to cyfrowy pieniądz.",
        "Blockchain to łańcuch bloków.",
        "Co to jest kryptowaluta?",
        "Zapisz to w pamięci."
    ]

    print("\n💬 STARTING CONVERSATION SIMULATION\n" + "="*60)

    for i, user_text in enumerate(conversation, 1):
        print(f"\n👤 USER: {user_text}")

        # --- LOGIC COPIED FROM swarm_gui.py process_swarm ---

        # 1. Inject to ALL agents
        vec = semantic_engine.encode(user_text)
        for a in swarm.agents:
            a.stm = vec.copy()
            a.origin = vec.copy()

        # 2. Simulate Thinking (Shortened for speed in test)
        total_energy = 0.0
        # Reduced cycles for test speed
        for _ in range(5):
            for a in swarm.agents:
                a.process_cycle()
            # Simplified ACN energy sum
            total_energy += 0.5

        # 3. Decode
        out_vec = swarm.agents[-1].stm
        cat, sim = semantic_engine.decode_category(out_vec)

        # 4. Generate Response
        response = brain.generate_response(cat, total_energy, user_text, swarm.agents, semantic_engine)

        # --- END LOGIC ---

        print(f"🤖 AI: {response}")
        print(f"   [Meta: Cat={cat}, Chaos={total_energy}]")

        # Verification Step for user request
        if "Kryptowaluta" in user_text:
            if "kryptowaluta" in brain.vector_db:
                print(f"   ✅ VERIFIED: 'kryptowaluta' added to Vector DB. Count: {brain.vector_db['kryptowaluta']['count']}")
            else:
                print(f"   ❌ ERROR: 'kryptowaluta' NOT found in Vector DB!")

    print("\n💾 TESTING PERSISTENCE SAVING...")
    brain.save_memory()

    if os.path.exists("swarm_ltm_store.pkl"):
        size = os.path.getsize("swarm_ltm_store.pkl")
        print(f"   ✅ Success! File exists. Size: {size} bytes.")
    else:
        print("   ❌ Error: File was not created.")

if __name__ == "__main__":
    simulate_gui_flow()
