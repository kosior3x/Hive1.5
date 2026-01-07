
import numpy as np
import time

class ChaoticNeuron:
    def __init__(self, neuron_id, r=3.99, x_init=None):
        """
        Inicjalizacja neuronu.

        Args:
            neuron_id: ID neuronu
            r: Parametr chaosu (dla mapy logistycznej r > 3.57 to chaos)
            x_init: Stan początkowy (0-1). Jeśli None, losowy.
        """
        self.id = neuron_id
        self.r = r
        self.x = x_init if x_init is not None else np.random.random()
        self.trajectory = []

    def step(self):
        """Wykonaj jeden krok mapy logistycznej: x_next = r * x * (1 - x)"""
        self.x = self.r * self.x * (1.0 - self.x)
        self.trajectory.append(self.x)
        return self.x

    def couple(self, external_signal, strength=0.1):
        """
        Sprzężenie z sygnałem zewnętrznym (inny neuron).
        Ta metoda "ciągnie" stan neuronu w stronę sygnału.
        """
        # Oblicz krok wewnętrzny
        internal_next = self.r * self.x * (1.0 - self.x)

        # Zastosuj sprzężenie (coupling)
        # x_new = f(x) + epsilon * (y - f(x))
        # Upraszczając: mieszanka własnej dynamiki i sygnału
        self.x = internal_next + strength * (external_signal - internal_next)

        # Zabezpieczenie zakresu [0,1] (rzadko potrzebne przy dobrym r, ale bezpieczne)
        self.x = np.clip(self.x, 0.0, 1.0)

        self.trajectory.append(self.x)
        return self.x

def run_experiment_sync():
    print(f"\n{'='*60}")
    print("🔬 EKSPERYMENT 1: SYNCHRONIZACJA CHAOSU")
    print(f"{'='*60}")

    # 1. Tworzymy dwa neurony z różnymi stanami początkowymi
    monitor_A = ChaoticNeuron(neuron_id="Master", r=3.9, x_init=0.2)
    follower_B = ChaoticNeuron(neuron_id="Slave ", r=3.9, x_init=0.8) # Zupełnie inny start

    print(f"Start A: {monitor_A.x:.4f}")
    print(f"Start B: {follower_B.x:.4f} (B musi dogonić A)")
    print("-" * 60)
    print(f"{'Step':<5} | {'Master A':<10} | {'Follower B':<10} | {'Error':<10} | {'Status'}")
    print("-" * 60)

    steps = 50
    coupling_strength = 0.4 # Siła połączenia

    for t in range(steps):
        # A robi krok (niezależnie)
        val_A = monitor_A.step()

        # B robi krok, ale widzi sygnał A (sprzężenie)
        # To symuluje "odbieranie" dynamiki, nie danych
        val_B = follower_B.couple(val_A, strength=coupling_strength)

        error = abs(val_A - val_B)

        status = "🔴 Chaos"
        if error < 0.1: status = "🟡 Converging"
        if error < 0.001: status = "🟢 SYNC!"

        print(f"{t+1:<5} | {val_A:.6f}   | {val_B:.6f}   | {error:.6f}   | {status}")

        # Symulacja opóźnienia, żeby wyglądało fajnie w terminalu
        # time.sleep(0.05)

    if error < 0.001:
        print("\n✅ SUKCES: Neuron B zsynchronizował się z Neuronem A.")
        print("To oznacza, że 'rozumieją' się bez przesyłania definicji.")
    else:
        print("\n❌ PORAŻKA: Brak synchronizacji. Zwiększ siłę sprzężenia.")

def run_experiment_signal():
    print(f"\n\n{'='*60}")
    print("🔬 EKSPERYMENT 2: KODOWANIE PRZEZ ZMIANĘ DYNAMIKI")
    print(f"{'='*60}")
    print("Scenariusz: Master A zmienia swoje 'r' (parametr chaosu).")
    print("Follower B wykrywa to jako 'Błąd synchronizacji', co jest sygnałem.")
    print("-" * 60)

    monitor_A = ChaoticNeuron(neuron_id="Master", r=3.8, x_init=0.3)
    follower_B = ChaoticNeuron(neuron_id="Slave ", r=3.8, x_init=0.31)

    # Najpierw synchronizacja
    for _ in range(20):
        val_A = monitor_A.step()
        follower_B.couple(val_A, strength=0.5)

    print("...Wstępna synchronizacja zakończona...")

    # TERAZ: Master A "chce coś powiedzieć" -> Zmienia swoją dynamikę
    print("\n📢 Master A zmienia parametr r (3.8 -> 3.95) [To jest sygnał!]")
    monitor_A.r = 3.95

    for t in range(15):
        val_A = monitor_A.step()
        # B próbuje nadążyć starym rytmem
        val_B = follower_B.couple(val_A, strength=0.5)

        error = abs(val_A - val_B)

        # Detekcja sygnału przez błąd
        detection = ""
        if error > 0.05:
            detection = "⚡ SYGNAŁ WYKRYTY!"

        print(f"Step {t+1:<2} | A(r={monitor_A.r}): {val_A:.4f} | B(r={follower_B.r}): {val_B:.4f} | Err: {error:.4f} {detection}")

if __name__ == "__main__":
    run_experiment_sync()
    run_experiment_signal()
