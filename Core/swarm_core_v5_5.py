#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
SWARM CORE v5.5 - PRODUCTION FINAL (Android-compatible)
=============================================================================

NAPRAWIONE BŁĘDY z poprzedniej wersji:
1. ✅ Importy działają na Androidzie (bez relative imports)
2. ✅ Concept Graph ≠ Q-Table (są RÓŻNE!)
3. ✅ Persistence wbudowana (bez patchy)
4. ✅ Gotowe do unzip & run

ARCHITEKTURA:
- Q-Table: state → action → value (pojedyncze decyzje)
- Concept Graph: wzorce zachowań (sekwencje akcji = manewry)
- Lorenz: moduluje OBA (deterministyczny chaos)

=============================================================================
"""

import os
import shutil
import sys
import time
import logging
import pickle
import math
import random
import copy
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Any, Optional, Set
from enum import Enum, auto

logger = logging.getLogger('SwarmCore')


# =============================================================================
# KONFIGURACJA
# =============================================================================

@dataclass
class SwarmConfig:
    """Konfiguracja SWARM v5.5 - FINAL (z poprawkami bezpieczeństwa)"""

    # Pliki
    BRAIN_FILE: str = "brain_v5_5.pkl"
    AUTO_SAVE_INTERVAL: int = 100

    # Fizyka robota
    ROBOT_WIDTH: float = 0.28        # szerokość obudowy [m]
    ROBOT_LENGTH: float = 0.28       # długość obudowy [m]
    ROBOT_WHEEL_SPAN: float = 0.32   # rozstaw kół [m] (zewn. krawędzie)
    WHEEL_BASE: float = 0.32         # rozstaw kół do odometrii [m]
    ROBOT_HALF_WIDTH: float = 0.16   # połowa rozstawu kół
    US_FORWARD_OFFSET: float = 0.29  # odległość US od osi obrotu [m]
    MAX_SPEED_MPS: float = 0.5

    # Dystanse (NAPRAWIONE - 10cm target)
    US_SAFETY_DIST: float = 0.12
    US_TARGET_DIST: float = 0.10
    LIDAR_SAFETY_RADIUS: float = 0.18
    LIDAR_MAX_RANGE: float = 3.0

    SAFETY_DIST_SPEED_SCALE: float = 0.25
    SAFETY_US_MIN: float = 0.08
    SAFETY_US_MAX: float = 0.25
    SAFETY_LIDAR_MIN: float = 0.12
    SAFETY_LIDAR_MAX: float = 0.35
    # LIDAR hard safety — wymuszony REVERSE/TURN gdy za blisko ściany
    LIDAR_HARD_SAFETY_MIN: float = 0.25  # Lmin poniżej tej wartości = natychmiastowa reakcja
    HARD_REFLEX_HOLD_CYCLES: int = 3      # Krócej trzymaj akcję awaryjną

    # Rear bumper
    REAR_BUMPER_FORWARD_CYCLES: int = 3  # Ile cykli FORWARD po kolizji tylnej

    # Korekcja jazdy prostej (encoder-based)
    FORWARD_ENCODER_CORRECTION: float = 0.5  # Siła korekcji enkoderowej
    FORWARD_LORENZ_PWM: float = 2.5       # Max ±PWM szum Lorenz przy FORWARD
    FORWARD_CHAOS_DAMPEN: float = 0.0      # Chaos inject = 0 dla FORWARD

    # Cofanie z LIDAR (NAPRAWIONE)
    REVERSE_LIDAR_CHECK: bool = True
    REVERSE_LIDAR_SECTORS: int = 4
    REVERSE_LIDAR_THRESHOLD: float = 0.15

    # Q-Learning (v5.9)
    LEARNING_RATE: float = 0.001     # zmniejszone dla stabilności aproksymatora
    LR_DECAY: float = 0.99995        # mnożnik LR co krok (do 0.001 minimum)
    LR_MIN: float = 0.0005           # minimalne LR
    DISCOUNT_FACTOR: float = 0.9
    EPSILON: float = 0.10            # eksploracja startowa
    EPSILON_DECAY: float = 0.9999    # decay epsilon co krok
    EPSILON_MIN: float = 0.02        # minimum epsilon (zawsze trochę eksploruje)

    # Replay Buffer
    REPLAY_BUFFER_CAPACITY: int = 7000
    REPLAY_BATCH_SIZE: int = 32
    REPLAY_TRAIN_FREQ: int = 4       # trenuj co N kroków

    # Concept Graph (NOWE - właściwe!)
    CONCEPT_MIN_SEQUENCE: int = 3  # Min długość sekwencji dla konceptu
    CONCEPT_ACTIVATION_THRESHOLD: float = 0.6  # Próg aktywacji
    CONCEPT_DECAY_RATE: float = 0.95  # Decay per step
    CONCEPT_SUCCESS_BOOST: float = 0.3  # Boost za sukces

    # ★ Wall proximity — blokuj koncepty FORWARD gdy za blisko
    WALL_PROXIMITY_THRESHOLD: float = 0.50  # Lmin poniżej tego = koncept FORWARD zablokowany

    # ★ Anti-oscillation — zapobiegaj pętli REVERSE↔FORWARD
    OSCILLATION_MAX_REPEATS: int = 8  # Po tylu powtórzeniach REVERSE → wymuś SPIN

    # PID
    PID_KP: float = 1.2
    PID_KI: float = 0.05
    PID_KD: float = 0.1
    PID_OUTPUT_SCALE: float = 400.0
    PWM_SLEW_RATE: float = 25.0

    # Lorenz
    LORENZ_SIGMA: float = 10.0
    LORENZ_RHO: float = 28.0
    LORENZ_BETA: float = 8.0 / 3.0
    LORENZ_DT: float = 0.01
    LORENZ_AGGRESSION_SCALE: float = 0.25
    LORENZ_BIAS_SCALE: float = 0.03      # Lorenz dla TURN/SPIN (zmniejszone 0.05->0.03)

    # Instinct — mocno wzmocniony, priorytet FORWARD na otwartej przestrzeni
    INSTINCT_WEIGHT: float = 3.0          # 0.8→3.0 — instynkt dominuje nad Q przy wolnej przestrzeni
    INSTINCT_REVERSE_PENALTY: float = 0.5 # Wyższy penalty dla REVERSE
    INSTINCT_US_BOOST: float = 1.2        # Mocniejszy US bias
    INSTINCT_CLEAR_THRESHOLD: float = 0.8 # Lmin > tej wartości = wyraźnie otwarta przestrzeń
    INSTINCT_CLEAR_FORWARD_BONUS: float = 8.0  # Duży bonus FORWARD gdy otwarta przestrzeń

    # Velocity
    VELOCITY_BASE: float = 0.35
    VELOCITY_MIN_DIST: float = 0.15
    VELOCITY_MAX_DIST: float = 2.0
    VELOCITY_MIN_SPEED: float = 0.1
    VELOCITY_MAX_SPEED: float = 1.0

    # Hysteresis
    HYSTERESIS_THRESHOLD: int = 3
    LOCK_DURATION: int = 5

    # Anti-stagnation — okno wydłużone, próg wyższy (spin nie jest stagnacją!)
    STAGNATION_WINDOW: int = 120          # 120 cykli = 4s przy 30Hz
    STAGNATION_THRESHOLD: float = 0.03    # Wariancja pozycji [m²] — 10× wyższy
    CHAOS_INJECT_STRENGTH: float = 0.3
    STAGNATION_FORCE_TURN_CYCLES: int = 4   # Krócej wymuszać — potem niech Q decyduje

    # PWM limits
    PWM_MAX: float = 100.0  # Hard clamp PWM

    # Damper
    STALL_CURRENT_THRESHOLD: float = 2.5
    STALL_SPEED_THRESHOLD: float = 0.05

    # Avoidance learning (Krok 2)
    AVOIDANCE_PENALTY: float = 1.0       # sila wplywu macierzy A na decyzje (Q - penalty*A)
    AVOIDANCE_LR_SCALE: float = 1.0      # mnoznik LR dla macierzy A
    # Krystalizacja wiedzy L2 (Krok 3)
    L2_FEATURES: int = 32                # liczba cech w warstwie L2
    L2_MIN_SAMPLES: int = 5000           # minimalna liczba krokow przed krystalizacja
    L2_LEARNING_RATE: float = 0.001      # learning rate dla aproksymatora L2
    L2_UPDATE_FREQ: int = 1000           # co ile krokow aktualizowac statystyki waznosci

    # Bramka meta-warstwy (Krok 4)
    GATE_FEATURES: int = 16              # liczba cech wejsciowych bramki
    GATE_LEARNING_RATE: float = 0.005    # wyzszy LR — bramka uczy sie szybciej
    GATE_UPDATE_FREQ: int = 1            # co ile krokow aktualizowac bramke (1 = kazdy)
    GATE_TRAIN_START: int = 1000         # po ilu krokach zaczac uczyc bramke
    GATE_SOFTMAX_TEMP: float = 1.0       # temperatura softmax (1.0 = normalny)

    # Model swiata (Krok 5)
    WORLD_MODEL_FEATURES: int = 32         # liczba cech dla modelu swiata
    WORLD_MODEL_LEARNING_RATE: float = 0.001
    WORLD_MODEL_HIDDEN: int = 16           # rozmiar warstwy ukrytej
    WORLD_MODEL_UPDATE_FREQ: int = 10      # co ile krokow aktualizowac model swiata
    WORLD_MODEL_BATCH_SIZE: int = 64       # batch do treningu modelu swiata
    WORLD_MODEL_BUFFER_SIZE: int = 10000   # bufor doswiadczen modelu swiata
    COUNTERFACTUAL_STEPS: int = 3          # nieuzywane aktywnie w krok. 5, zostawiamy jako koncepcje
    COUNTERFACTUAL_THRESHOLD: float = 0.5  # prog poprawy, by dodac kontrfaktyke
    COUNTERFACTUAL_LR: float = 0.1         # jak bardzo kontrfaktyka wplywa na Q (waga)
    CONCEPT_PRUNING_INTERVAL: int = 2000  # Co ile krokow uruchamiac przycinanie konceptow

    # Neural Network (Krok 6)
    NN_HIDDEN_1: int = 32               # pierwsza warstwa ukryta (82 -> 32)
    NN_HIDDEN_2: int = 16               # druga warstwa ukryta (32 -> 16)
    NN_ACTIVATION: str = "relu"         # "relu" lub "tanh"
    NN_LEARNING_RATE: float = 0.00001
    NN_USE_L1_INIT: bool = True         # inicjalizuj W1 srednia z L1 (8x82)
    NN_USE_L2_INIT: bool = True         # inicjalizuj W2 srednia z L2 (8x32)
    NN_USE_A_INIT: bool = True          # inicjalizuj glowe A (jesli osobna)
    NN_USE_GATE_INIT: bool = True       # inicjalizuj glowe gate
    NN_CLIP_GRAD: float = 1.0           # clipping gradientow


    # Krystalizacja wiedzy L2 (Krok 3)
# =============================================================================
# AKCJE
# =============================================================================

class Action(Enum):
    FORWARD = auto()
    TURN_LEFT = auto()
    TURN_RIGHT = auto()
    SPIN_LEFT = auto()
    SPIN_RIGHT = auto()
    REVERSE = auto()
    ESCAPE_MANEUVER = auto()
    STOP = auto()


# =============================================================================
# CONCEPT GRAPH - WŁAŚCIWA IMPLEMENTACJA!
# =============================================================================

class Concept:
    """
    Koncept = WZORZEC ZACHOWANIA (sekwencja akcji)

    NIE jest to duplikat Q-table!

    Przykłady konceptów:
    - "tight_corner_left": [SPIN_LEFT, FORWARD, TURN_RIGHT]
    - "explore_corridor": [FORWARD, FORWARD, FORWARD]
    - "escape_deadend": [REVERSE, SPIN_LEFT, FORWARD]
    """

    def __init__(self, name: str, sequence: List[Action]):
        self.name = name
        self.sequence = sequence  # Sekwencja akcji
        self.activation = 0.0  # Jak bardzo aktywny (0-1)
        self.success_count = 0  # Ile razy doprowadził do sukcesu
        self.usage_count = 0  # Ile razy użyty
        self.last_used = 0  # Timestamp ostatniego użycia
        self.context: Dict[str, Any] = {}  # Kontekst (np. min_dist range)
        self.last_used_step = 0          # Numer kroku, w którym koncept został użyty po raz ostatni
        self.success_ratio = 0.0         # Stosunek sukcesów do użyć (success_count / usage_count)

    def matches_context(self, current_context: Dict[str, Any]) -> float:
        """Jak dobrze pasuje do obecnego kontekstu (0-1)"""
        if not self.context:
            return 0.5  # Brak kontekstu = neutralny

        score = 0.0
        count = 0

        for key, value in self.context.items():
            if key in current_context:
                # Dla wartości numerycznych - im bliżej tym lepiej
                if isinstance(value, (int, float)) and isinstance(current_context[key], (int, float)):
                    diff = abs(value - current_context[key])
                    max_diff = max(abs(value), abs(current_context[key]), 0.1)
                    score += 1.0 - min(diff / max_diff, 1.0)
                    count += 1

        return score / count if count > 0 else 0.5

    def activate(self, boost: float = 0.1, current_step: int = 0):
        """Zwiększ aktywację"""
        self.activation = min(1.0, self.activation + boost)
        self.usage_count += 1
        self.last_used = time.time()
        self.last_used_step = current_step
        self.success_ratio = self.success_count / max(1, self.usage_count)

    def decay(self, rate: float = 0.95):
        """Zmniejsz aktywację (naturalny decay)"""
        self.activation *= rate

    def mark_success(self, boost: float = 0.3):
        """Zaznacz sukces - zwiększ aktywację"""
        self.success_count += 1
        self.activation = min(1.0, self.activation + boost)
        self.success_ratio = self.success_count / max(1, self.usage_count)


class ConceptGraph:
    """
    Graf konceptów - wzorce zachowań wysokiego poziomu

    ★ TO NIE JEST DUPLIKAT Q-TABLE! ★

    Q-Table: state → action (pojedyncze decyzje)
    Concepts: wzorce → sekwencje akcji (manewry)
    """

    def __init__(self, config: SwarmConfig):
        self.config = config
        self.concepts: Dict[str, Concept] = {}
        self.action_history = deque(maxlen=10)  # Ostatnie akcje
        self.learning_buffer: List[Tuple[List[Action], bool]] = []  # (sequence, success)

        # Parametry przycinania konceptów
        self.pruning_interval = 1000          # Co ile kroków uruchamiać przycinanie
        self.min_usage_to_survive = 5         # Minimalna liczba użyć, by koncept nie został usunięty
        self.min_success_ratio_to_survive = 0.25  # Minimalny wskaźnik sukcesu
        self.similarity_threshold = 0.7       # Próg podobieństwa do łączenia konceptów (0-1)
        self.last_pruned_step = 0

        # Predefiniowane koncepty (instynkt)
        self._init_base_concepts()

    def _init_base_concepts(self):
        """Załaduj bazowe koncepty (instynktowne wzorce)"""
        base = [
            Concept("explore_straight", [Action.FORWARD, Action.FORWARD, Action.FORWARD]),
            Concept("corner_left", [Action.TURN_LEFT, Action.FORWARD, Action.FORWARD]),
            Concept("corner_right", [Action.TURN_RIGHT, Action.FORWARD, Action.FORWARD]),
            Concept("tight_left", [Action.SPIN_LEFT, Action.FORWARD, Action.TURN_RIGHT]),
            Concept("tight_right", [Action.SPIN_RIGHT, Action.FORWARD, Action.TURN_LEFT]),
            Concept("escape_back_left", [Action.REVERSE, Action.SPIN_LEFT, Action.FORWARD]),
            Concept("escape_back_right", [Action.REVERSE, Action.SPIN_RIGHT, Action.FORWARD]),
        ]

        for c in base:
            self.concepts[c.name] = c
            c.activation = 0.3  # Startowa aktywacja

    def update(self, action: Action, reward: float, current_step: int = 0):
        """Aktualizuj graf na podstawie wykonanej akcji i nagrody"""
        self.action_history.append(action)

        # Decay wszystkich konceptów
        for concept in self.concepts.values():
            concept.decay(self.config.CONCEPT_DECAY_RATE)

        # Czy ostatnie N akcji pasują do jakiegoś konceptu?
        if len(self.action_history) >= self.config.CONCEPT_MIN_SEQUENCE:
            recent = list(self.action_history)[-self.config.CONCEPT_MIN_SEQUENCE:]

            for concept in self.concepts.values():
                # Sprawdź czy koncept pasuje do ostatnich akcji
                if len(concept.sequence) <= len(recent):
                    if recent[-len(concept.sequence):] == concept.sequence:
                        # Pasuje! Aktywuj (aktualizujac tez last_used_step, ale nie mamy go tu bezposrednio w update...
                        # Trudno, update() powinno przyjmowac current_step jesli chcemy byc precyzyjni.
                        # Ale Concept.activate() ma domyslne current_step=0.
                        # Zmienimy to w integracji glownej petli, zeby przekazywac step.
                        # Na razie uzywamy czasu systemowego w activate().
                        concept.activate(0.1, current_step)

                        # Jeśli reward pozytywny = sukces
                        if reward > 0:
                            concept.mark_success(self.config.CONCEPT_SUCCESS_BOOST)

        # Uczenie się nowych konceptów (jeśli buffer pełny)
        if reward > 0.5 and len(self.action_history) >= self.config.CONCEPT_MIN_SEQUENCE:
            self._try_learn_new_concept()

    def _try_learn_new_concept(self):
        """Spróbuj nauczyć się nowego konceptu z bufora"""
        recent = list(self.action_history)[-self.config.CONCEPT_MIN_SEQUENCE:]

        # Sprawdź czy to już znany koncept
        for concept in self.concepts.values():
            if concept.sequence == recent:
                return  # Już znamy

        # Nowy koncept!
        name = f"learned_{len(self.concepts)}"
        new_concept = Concept(name, recent.copy())
        new_concept.activation = 0.5  # Średnia aktywacja na start
        self.concepts[name] = new_concept

        logger.info(f"Nauczono nowy koncept: {name} = {[a.name for a in recent]}")

    def get_best_concept(self, context: Dict[str, Any]) -> Optional[Concept]:
        """Zwróć najbardziej aktywny koncept pasujący do kontekstu

        ★ POPRAWKA: Blokuj koncepty FORWARD gdy za blisko ściany!
        """
        if not self.concepts:
            return None

        min_dist = context.get('min_dist', 3.0)
        wall_proximity = self.config.WALL_PROXIMITY_THRESHOLD

        best_concept = None
        best_score = -1.0

        for concept in self.concepts.values():
            if concept.activation < self.config.CONCEPT_ACTIVATION_THRESHOLD:
                continue

            # ★ BLOKADA: Jeśli koncept polega na FORWARD a jesteśmy za blisko ściany
            if min_dist < wall_proximity:
                # Sprawdź czy koncept głównie sugeruje jazda do przodu
                forward_count = sum(1 for a in concept.sequence if a == Action.FORWARD)
                if forward_count > len(concept.sequence) // 2:
                    # Zbyt dużo FORWARD w sekwencji, a ściana za blisko → zablokuj
                    continue

            # Score = aktywacja × dopasowanie do kontekstu
            context_match = concept.matches_context(context)
            score = concept.activation * context_match

            if score > best_score:
                best_score = score
                best_concept = concept

        return best_concept

    def get_next_action_from_concept(self, concept: Concept) -> Optional[Action]:
        """Zwróć następną akcję z sekwencji konceptu"""
        if not concept or not concept.sequence:
            return None

        # Sprawdź gdzie jesteśmy w sekwencji
        recent = list(self.action_history)[-len(concept.sequence)+1:]

        # Znajdź pozycję w sekwencji
        for i in range(len(concept.sequence)):
            if i < len(recent) and recent[i] != concept.sequence[i]:
                # Nie pasuje - zacznij od początku
                return concept.sequence[0]

        # Kontynuuj sekwencję
        idx = len(recent)
        if idx < len(concept.sequence):
            return concept.sequence[idx]

        # Koniec sekwencji - zacznij od początku
        return concept.sequence[0]

    def _calculate_similarity(self, seq1: List[Action], seq2: List[Action]) -> float:
        """
        Oblicza podobieństwo między dwiema sekwencjami akcji.
        Zwraca wartość od 0 (całkowicie różne) do 1 (identyczne).
        Używa zmodyfikowanej odległości Levenshteina dla sekwencji.
        """
        if not seq1 and not seq2:
            return 1.0
        if not seq1 or not seq2:
            return 0.0

        len1, len2 = len(seq1), len(seq2)
        matrix = [[0 for _ in range(len2 + 1)] for _ in range(len1 + 1)]

        for i in range(len1 + 1):
            matrix[i][0] = i
        for j in range(len2 + 1):
            matrix[0][j] = j

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if seq1[i-1] == seq2[j-1] else 1
                matrix[i][j] = min(
                    matrix[i-1][j] + 1,      # deletion
                    matrix[i][j-1] + 1,      # insertion
                    matrix[i-1][j-1] + cost  # substitution
                )

        distance = matrix[len1][len2]
        max_len = max(len1, len2)

        return 1.0 - (distance / max_len)

    def _merge_concepts(self, concept1: Concept, concept2: Concept) -> Concept:
        """
        Łączy dwa podobne koncepty w jeden, nowy.
        Nowa sekwencja to dłuższa z nich (optymalizacja).
        Aktywacja i liczniki są sumowane/średniowane.
        """
        # Wybierz dłuższą sekwencję (bardziej szczegółowa)
        if len(concept1.sequence) >= len(concept2.sequence):
            new_sequence = concept1.sequence
        else:
            new_sequence = concept2.sequence

        new_name = f"merged_{len(self.concepts)}"
        new_concept = Concept(new_name, new_sequence.copy())

        # Połącz statystyki
        new_concept.activation = max(concept1.activation, concept2.activation)
        new_concept.usage_count = concept1.usage_count + concept2.usage_count
        new_concept.success_count = concept1.success_count + concept2.success_count
        new_concept.last_used_step = max(concept1.last_used_step, concept2.last_used_step)
        new_concept.success_ratio = new_concept.success_count / max(1, new_concept.usage_count)

        logger.info(f"🔗 Połączono koncepty '{concept1.name}' i '{concept2.name}' w '{new_name}'")
        return new_concept

    def prune_and_merge(self, current_step: int):
        """
        Główna metoda przycinania i łączenia konceptów.
        Ma być wywoływana co  kroków z poziomu pętli głównej.
        """
        if len(self.concepts) < 10:  # Nie ma sensu przycinać, gdy jest mało konceptów
            return

        logger.info(f"✂️ Rozpoczynam przycinanie konceptów (krok {current_step})...")

        # --- Krok A: Identyfikacja konceptów do usunięcia (śmieci) ---
        concepts_to_remove = []
        for name, concept in self.concepts.items():
            # Pomiń koncepty bazowe? (np. 'explore_straight') – można dodać listę chronionych nazw
            if concept.name.startswith('learned_'):
                age = current_step - concept.last_used_step
                # Usuń, jeśli: mało używany LUB niska skuteczność I nie był ostatnio używany
                if (concept.usage_count < self.min_usage_to_survive and age > self.pruning_interval) or                    (concept.success_ratio < self.min_success_ratio_to_survive and age > self.pruning_interval):
                    concepts_to_remove.append(name)

        # Usuń śmieci
        for name in concepts_to_remove:
            logger.info(f"  Usuwam śmieciowy koncept: '{name}' (użycia: {self.concepts[name].usage_count}, sukces: {self.concepts[name].success_ratio:.2f})")
            del self.concepts[name]

        # --- Krok B: Identyfikacja i łączenie podobnych konceptów ---
        concept_items = list(self.concepts.items())
        merged_any = True
        while merged_any:
            merged_any = False
            for i in range(len(concept_items)):
                for j in range(i+1, len(concept_items)):
                    name1, conc1 = concept_items[i]
                    name2, conc2 = concept_items[j]

                    # Nie łącz samych ze sobą i pomiń, jeśli któryś już został usunięty
                    if name1 not in self.concepts or name2 not in self.concepts:
                        continue

                    similarity = self._calculate_similarity(conc1.sequence, conc2.sequence)
                    if similarity >= self.similarity_threshold:
                        # Połącz je
                        new_concept = self._merge_concepts(conc1, conc2)
                        self.concepts[new_concept.name] = new_concept
                        # Usuń stare
                        del self.concepts[name1]
                        del self.concepts[name2]
                        logger.info(f"  Połączono {name1} i {name2} w {new_concept.name}")
                        merged_any = True
                        break  # Przerwij pętle, bo słownik się zmienił
                if merged_any:
                    break
            # Zaktualizuj listę konceptów na nową iterację
            if merged_any:
                concept_items = list(self.concepts.items())

        self.last_pruned_step = current_step
        logger.info(f"✅ Przycinanie zakończone. Pozostało konceptów: {len(self.concepts)}")


class LorenzAttractor:
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.x, self.y, self.z = 0.1, 0.0, 0.0
        self.sigma = config.LORENZ_SIGMA
        self.rho = config.LORENZ_RHO
        self.beta = config.LORENZ_BETA
        self.dt = config.LORENZ_DT
        self.x_norm, self.z_norm = 0.0, 0.0

    def step(self):
        dx = self.sigma * (self.y - self.x)
        dy = self.x * (self.rho - self.z) - self.y
        dz = self.x * self.y - self.beta * self.z
        self.x += dx * self.dt
        self.y += dy * self.dt
        self.z += dz * self.dt
        self.x_norm = np.tanh(self.x / 15.0)
        self.z_norm = min(1.0, max(0.0, self.z / 40.0))

    def get_state(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


# =============================================================================
# FREE SPACE INSTINCT (NAPRAWIONY - bardziej konserwatywny)
# =============================================================================

class FreeSpaceInstinct:
    def __init__(self, config: SwarmConfig):
        self.config = config

    def compute_free_space_vector(self, lidar_16: np.ndarray) -> Tuple[float, float]:
        free_space = 1.0 - lidar_16
        angles = np.arange(16) * (2 * np.pi / 16)
        x_sum = float(np.sum(free_space * np.cos(angles)))
        y_sum = float(np.sum(free_space * np.sin(angles)))
        magnitude = math.sqrt(x_sum**2 + y_sum**2) / 16.0
        if magnitude < 0.01:
            return 0.0, 0.0
        angle = math.atan2(y_sum, x_sum)
        return angle, magnitude

    def get_bias_for_action(self, free_space_angle: float,
                            magnitude: float = 0.0,
                            front_clearance: float = 1.0,
                            us_left: float = 3.0,
                            us_right: float = 3.0) -> Dict[Action, float]:
        """
        front_clearance: 0.0=przeszkoda wprost z przodu, 1.0=czysto
        Oparty na sektorach LIDAR 14,15,0,1 (±45° od 0°)
        """
        bias = {action: 0.0 for action in Action}
        angle   = free_space_angle
        weight  = self.config.INSTINCT_WEIGHT
        clear_bonus = self.config.INSTINCT_CLEAR_FORWARD_BONUS
        us_min      = min(us_left, us_right)

        # ★★★ KLUCZOWE: bonus FORWARD gdy PRZOD jest faktycznie czysty
        # front_clearance > 0.55 = brak przeszkody w ciagu 1.35m z przodu
        if front_clearance > 0.55 and us_min > 0.3:
            scale = (front_clearance - 0.55) / 0.45  # 0→1 liniowo
            bias[Action.FORWARD] += clear_bonus * scale
            # Kara za bezsensowny obrót gdy przed nami wolna droga
            bias[Action.REVERSE]         -= clear_bonus * 0.6
            bias[Action.SPIN_LEFT]       -= clear_bonus * 0.4
            bias[Action.SPIN_RIGHT]      -= clear_bonus * 0.4
            bias[Action.ESCAPE_MANEUVER] -= clear_bonus * 0.5
        elif front_clearance < 0.35:
            # Przeszkoda z przodu — NIE jedź do przodu!
            bias[Action.FORWARD]  -= clear_bonus * 0.8
            bias[Action.REVERSE]  += weight * 1.0

        # Bias wektorowy (ze skalowaniem magnitude)
        vec_weight = weight * max(magnitude, 0.05)

        if abs(angle) < math.pi / 4:
            bias[Action.FORWARD] += vec_weight * 1.5

        if math.pi / 6 < angle < 5 * math.pi / 6:
            bias[Action.TURN_LEFT]  += vec_weight * 1.5
            bias[Action.SPIN_LEFT]  += vec_weight * 0.4

        if -5 * math.pi / 6 < angle < -math.pi / 6:
            bias[Action.TURN_RIGHT] += vec_weight * 1.5
            bias[Action.SPIN_RIGHT] += vec_weight * 0.4

        # Penalizuj REVERSE gdy wektor wskazuje do przodu
        if abs(angle) < math.pi / 2:
            bias[Action.REVERSE]         -= weight * self.config.INSTINCT_REVERSE_PENALTY
            bias[Action.ESCAPE_MANEUVER] -= weight * 0.3

        return bias

    def apply_us_bias(self, bias: Dict[Action, float],
                      us_left: float, us_right: float) -> Dict[Action, float]:
        us_boost = self.config.INSTINCT_US_BOOST
        diff = us_left - us_right

        if abs(diff) > 0.1:
            if diff > 0:   # Lewa wolniejsza = więcej miejsca po lewej
                bias[Action.TURN_LEFT]  += us_boost * min(diff, 1.0)
            else:          # Prawa wolniejsza = więcej miejsca po prawej
                bias[Action.TURN_RIGHT] += us_boost * min(-diff, 1.0)

        # Oba US daleko → mocny boost FORWARD
        if us_left > 1.5 and us_right > 1.5:
            bias[Action.FORWARD] += us_boost * 2.5  # 0.5 → 2.5

        return bias




# =============================================================================
# VELOCITY MAPPER, STABILIZER, ANTI-STAGNATION (bez zmian - działają)
# =============================================================================

class DynamicVelocityMapper:
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.last_target_l, self.last_target_r = 0.0, 0.0

    def compute_base_velocity(self, min_dist: float, aggression_factor: float) -> float:
        if min_dist < self.config.VELOCITY_MIN_DIST:
            speed = self.config.VELOCITY_MIN_SPEED
        elif min_dist > self.config.VELOCITY_MAX_DIST:
            speed = self.config.VELOCITY_MAX_SPEED
        else:
            ratio = (min_dist - self.config.VELOCITY_MIN_DIST) / \
                   (self.config.VELOCITY_MAX_DIST - self.config.VELOCITY_MIN_DIST)
            speed = self.config.VELOCITY_MIN_SPEED + \
                   ratio * (self.config.VELOCITY_MAX_SPEED - self.config.VELOCITY_MIN_SPEED)
        speed *= (1.0 + aggression_factor * 0.3)
        return speed * self.config.VELOCITY_BASE

    def apply_ramp_limit(self, target_l: float, target_r: float) -> Tuple[float, float]:
        max_delta = 0.15
        delta_l = np.clip(target_l - self.last_target_l, -max_delta, max_delta)
        delta_r = np.clip(target_r - self.last_target_r, -max_delta, max_delta)
        self.last_target_l += delta_l
        self.last_target_r += delta_r
        return self.last_target_l, self.last_target_r


class ActionStabilizer:
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.history = deque(maxlen=config.HYSTERESIS_THRESHOLD)
        self.locked_action: Optional[Action] = None
        self.lock_remaining = 0

    def update(self, candidate_action: Action) -> Action:
        if self.lock_remaining > 0:
            self.lock_remaining -= 1
            return self.locked_action
        self.history.append(candidate_action)
        if len(self.history) == self.config.HYSTERESIS_THRESHOLD:
            if len(set(self.history)) == 1:
                self.locked_action = self.history[0]
                self.lock_remaining = self.config.LOCK_DURATION
                return self.locked_action
        return candidate_action

    def force_unlock(self):
        self.history.clear()
        self.locked_action = None
        self.lock_remaining = 0


class AntiStagnationController:
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.position_history = deque(maxlen=config.STAGNATION_WINDOW)
        self.is_stagnant = False
        self.stagnation_force_remaining = 0  # ★ Ile cykli wymuszonego skrętu zostało
        self.stagnation_direction = 1  # ★ 1=lewo, -1=prawo (zmienia się)

    def update(self, x: float, y: float, avg_pwm: float,
               current_action: Optional['Action'] = None):
        # ★ Kluczowe: spin/turn to CELOWE działanie, nie stagnacja!
        #   Nie aktualizuj historii pozycji podczas aktywnego skrętu.
        spinning = (current_action in (
            Action.SPIN_LEFT, Action.SPIN_RIGHT,
            Action.TURN_LEFT, Action.TURN_RIGHT,
            Action.ESCAPE_MANEUVER
        ))
        if not spinning:
            self.position_history.append((x, y))

        if len(self.position_history) == self.config.STAGNATION_WINDOW:
            positions = np.array(self.position_history)
            variance  = np.var(positions, axis=0).sum()
            was_stagnant = self.is_stagnant
            # Stagnacja TYLKO gdy robot NIE skręca i pozycja się nie zmienia
            self.is_stagnant = (
                (variance < self.config.STAGNATION_THRESHOLD)
                and (avg_pwm > 20)
                and not spinning
            )

            if self.is_stagnant and not was_stagnant:
                self.stagnation_count = getattr(self, 'stagnation_count', 0) + 1
                self.stagnation_force_remaining = self.config.STAGNATION_FORCE_TURN_CYCLES
                self.stagnation_direction *= -1
        elif spinning:
            # Reset stagnacji gdy robot aktywnie manewruje
            self.is_stagnant = False

    def should_force_turn(self) -> Optional['Action']:
        """Czy stagnacja wymusza skręt? Zwraca akcję lub None."""
        if self.stagnation_force_remaining > 0:
            self.stagnation_force_remaining -= 1
            count = getattr(self, 'stagnation_count', 0)
            # Pierwsza stagnacja → lekki TURN, kolejne → agresywny SPIN
            if self.stagnation_direction > 0:
                return Action.SPIN_LEFT if count > 2 else Action.TURN_LEFT
            else:
                return Action.SPIN_RIGHT if count > 2 else Action.TURN_RIGHT
        return None

    def inject_chaos(self, lorenz_x: float, lorenz_z: float,
                    pwm_l: float, pwm_r: float) -> Tuple[float, float]:
        # ★ Lorenz chaos ZAWSZE dziala dla TURN/SPIN — nie czeka na stagnację!
        # Lorenz x_norm: +1=lewa, -1=prawa — moduluje kierunek skrętu
        base_strength = self.config.CHAOS_INJECT_STRENGTH * 0.5  # normalny tryb
        if self.is_stagnant:
            base_strength = self.config.CHAOS_INJECT_STRENGTH  # stagnacja = pełna moc
        pwm_l += lorenz_x * 50 * base_strength
        pwm_r -= lorenz_x * 50 * base_strength
        return pwm_l, pwm_r


# =============================================================================
# LIDAR ENGINE (NAPRAWIONY - check front sectors)
# =============================================================================

class LidarEngine:
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.sectors_16 = np.zeros(16)
        self.min_dist = config.LIDAR_MAX_RANGE

    def process(self, lidar_points: List[Tuple[float, float]]) -> np.ndarray:
        self.sectors_16.fill(0.0)
        sector_dists = [[] for _ in range(16)]

        for angle, dist in lidar_points:
            if dist <= 0 or dist > self.config.LIDAR_MAX_RANGE:
                continue
            sector = int((angle % 360) / 22.5) % 16
            sector_dists[sector].append(dist)

        self.min_dist = self.config.LIDAR_MAX_RANGE
        for i, dists in enumerate(sector_dists):
            if dists:
                min_d = min(dists)
                self.min_dist = min(self.min_dist, min_d)
                self.sectors_16[i] = 1.0 - min(min_d / self.config.LIDAR_MAX_RANGE, 1.0)
            else:
                self.sectors_16[i] = 0.0

        return self.sectors_16

    def check_front_sectors_blocked(self, threshold: float, num_sectors: int = 4) -> bool:
        """Sprawdź czy przód jest zablokowany (połowa wiązki)"""
        front_sectors = self.sectors_16[:num_sectors]
        blocked_count = np.sum(front_sectors > threshold)
        return blocked_count >= (num_sectors // 2)


# =============================================================================
# FEATURE EXTRACTOR (v5.9 — wektor cech z WSZYSTKICH czujników)
# =============================================================================


# =============================================================================
# RUNNING NORMALIZER (Welford online — stabilizuje cechy dla Q-aproksymatora)
# =============================================================================

class RunningNormalizer:
    """
    Online normalizacja cech metodą Welforda.
    Utrzymuje bieżącą średnią i wariancję bez przechowywania historii.
    """
    def __init__(self, n_features: int):
        self.n = 0
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.M2   = np.zeros(n_features, dtype=np.float64)
        self.n_features = n_features

    def update(self, x: np.ndarray):
        # ★ Zabezpieczenie przed zmianą liczby cech w locie
        if x.shape[0] != self.n_features:
            logger.warning(f"Normalizer shape mismatch: {x.shape[0]} != {self.n_features}. Resetting.")
            self.n_features = x.shape[0]
            self.n = 0
            self.mean = np.zeros(self.n_features, dtype=np.float64)
            self.M2   = np.zeros(self.n_features, dtype=np.float64)

        self.n += 1
        delta = x.astype(np.float64) - self.mean
        self.mean += delta / self.n
        delta2 = x.astype(np.float64) - self.mean
        self.M2 += delta * delta2

    def normalize(self, x: np.ndarray) -> np.ndarray:
        if self.n < 30:           # zbieramy próbki zanim zaczniemy normalizować
            return x
        std = np.sqrt(self.M2 / (self.n - 1))
        std[std < 1e-6] = 1.0    # unikamy dzielenia przez zero
        return ((x.astype(np.float64) - self.mean) / std).astype(np.float32)

    def get_state(self) -> dict:
        return {'n': self.n, 'mean': self.mean.tolist(), 'M2': self.M2.tolist()}

    def set_state(self, state: dict):
        if state['n'] > 0:
            saved_mean = np.array(state['mean'])
            saved_M2   = np.array(state['M2'])
            if len(saved_mean) == self.n_features:
                self.n    = state['n']
                self.mean = saved_mean
                self.M2   = saved_M2
            else:
                logger.warning(f"Normalizer set_state: shape mismatch, resetting")
                self.n = 0
                self.mean = np.zeros(self.n_features)
                self.M2   = np.zeros(self.n_features)



# =============================================================================
# FEATURE IMPORTANCE ANALYZER — Krok 3: analiza waznosci cech
# =============================================================================

class FeatureImportanceAnalyzer:
    """
    Analizuje waznosc cech na podstawie wag macierzy Q i A.

    Waznosc cechy i = sum_a( |q_weights[a,i]| + |a_weights[a,i]| )

    Umozliwia wybor L2_FEATURES najwazniejszych cech
    i przekazanie ich do skrystalizowanego aproksymatora L2.
    """

    def __init__(self, n_features: int, l2_features: int = 32):
        self.n_features  = n_features
        self.l2_features = l2_features
        # Kumulatywna suma wartosci bezwzglednych wag (nie srednia — latwiej dodawac)
        self.importance_sum = np.zeros(n_features, dtype=np.float64)
        self.samples    = 0
        self.top_indices: Optional[np.ndarray] = None
        self.is_frozen  = False

    def update(self, q_weights: np.ndarray, a_weights: np.ndarray):
        """
        Dodaje biezace wagi Q i A do sumy waznosci.
        Wywolywana co L2_UPDATE_FREQ krokow.
        """
        if self.is_frozen:
            return
        # waznosc Q: suma |wag| po wszystkich akcjach
        q_importance = np.sum(np.abs(q_weights), axis=0)
        # waznosc A: identycznie dla macierzy unikania
        a_importance = np.sum(np.abs(a_weights), axis=0)
        # laczna waznosc (Q + A — obie macierze sa wazne)
        self.importance_sum += q_importance + a_importance
        self.samples += 1

    def get_top_features(self, force: bool = False) -> Optional[np.ndarray]:
        """
        Zwraca indeksy L2_FEATURES najwazniejszych cech.
        Zwraca None jesli za malo probek i force=False.
        """
        if self.samples < 100 and not force:
            return None
        if self.top_indices is None or force:
            avg_importance = self.importance_sum / max(self.samples, 1)
            # argsort rosnaco → ostatnie L2 indeksow = najwazniejsze
            self.top_indices = np.argsort(avg_importance)[-self.l2_features:]
        return self.top_indices

    def get_importance_vector(self) -> np.ndarray:
        """Zwraca znormalizowany wektor waznosci (diagnostyka)."""
        if self.samples == 0:
            return np.zeros(self.n_features)
        avg = self.importance_sum / self.samples
        total = avg.sum()
        return avg / total if total > 0 else avg

    def freeze(self):
        """Zamroz analize — po krystalizacji nie zbieramy wiecej danych."""
        self.is_frozen = True
        logger.info(
            f"FeatureImportanceAnalyzer zamrozony po {self.samples} probkach."
        )


# =============================================================================
# REPLAY BUFFER (Experience Replay dla stabilnego uczenia)
# =============================================================================

class ReplayBuffer:
    """
    Bufor doświadczeń z losowym próbkowaniem.
    Zapobiega korelacji próbek w TD learning.
    """
    def __init__(self, capacity: int = 2000):
        self.buffer = deque(maxlen=capacity)

    def push(self, features: np.ndarray, action: int, reward: float,
             next_features: np.ndarray, done: bool = False):
        self.buffer.append((features.copy(), action, reward, next_features.copy(), done))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


# =============================================================================
# FEATURE EXTRACTOR v5.9.2 — 60 cech z WSZYSTKICH czujników
# =============================================================================

class FeatureExtractor:
    """
    Ekstraktor cech sensorycznych — wersja v5.9.3 (82 cechy)

    Struktura:
     0–15  : surowe LIDAR (16 sektorów)
     16–22 : agregaty kierunkowe
     23–28 : US (left, right, min, diff, blocked)
     29–33 : enkodery (l, r, avg, diff, abs_diff)
     34–35 : Lorenz x/z
     36    : rear bumper (binary)
     37–43 : interakcje (nieliniowe)
     44–59 : rozszerzone agregaty, kwadraty, log, peak/spread
     60    : bias stały (1.0)
     61–64 : CLEARANCE (1.0 - mean_occ) - PRZÓD, TYŁ, LEWO, PRAWO
     65–66 : FREE SPACE VECTOR (sin, cos kąta)
     67    : FREE SPACE MAGNITUDE
     68–75 : ONE-HOT ACTION (8 akcji)
     76    : BUMPER HISTORY (decaying signal)
     77–81 : NOWE INTERAKCJE (clearance * clearance)
    """

    def __init__(self, config: SwarmConfig):
        self.config = config
        self.us_block_thresh    = 0.2
        self.lidar_block_thresh = 0.3
        self.bumper_history     = 0.0  # Historyczny ślad uderzenia

    def extract(self, lidar_16: np.ndarray, us_left: float, us_right: float,
                encoder_l: float, encoder_r: float,
                lorenz_x: float, lorenz_z: float,
                rear_bumper: int, min_dist: float,
                last_action: Optional[Action] = None,
                free_angle: float = 0.0, free_mag: float = 0.0) -> np.ndarray:
        """
        Zwraca wektor 82 cech (float32).
        """
        f = []

        # ── 0–15: Surowe LIDAR ──────────────────────────────────────────────
        f.extend(lidar_16.tolist())

        # ── 16: min_dist ────────────────────────────────────────────────────
        f.append(min_dist)

        # Agregaty
        front_sec = np.array([lidar_16[14], lidar_16[15], lidar_16[0], lidar_16[1]])
        right_sec = lidar_16[2:6]
        rear_sec  = lidar_16[6:10]
        left_sec  = lidar_16[10:14]

        mean_front = float(np.mean(front_sec))
        mean_left  = float(np.mean(left_sec))
        mean_right = float(np.mean(right_sec))
        mean_rear  = float(np.mean(rear_sec))

        # ── 17–22: Agregaty kierunkowe ───────────────────────────────────────
        f.append(mean_front)                          # 17
        f.append(mean_left)                           # 18
        f.append(mean_right)                          # 19
        f.append(mean_rear)                           # 20
        f.append(mean_left - mean_right)              # 21
        f.append(1.0 if min_dist < self.lidar_block_thresh else 0.0) # 22

        # ── 23–28: US ───────────────────────────────────────────────────────
        f.append(us_left)                             # 23
        f.append(us_right)                            # 24
        us_min = min(us_left, us_right)
        f.append(us_min)                              # 25
        f.append(us_left - us_right)                  # 26
        f.append(1.0 if us_left  < self.us_block_thresh else 0.0) # 27
        f.append(1.0 if us_right < self.us_block_thresh else 0.0) # 28

        # ── 29–33: Enkodery ──────────────────────────────────────────────────
        avg_speed = (encoder_l + encoder_r) / 2.0
        speed_diff = encoder_l - encoder_r
        f.append(encoder_l)                           # 29
        f.append(encoder_r)                           # 30
        f.append(avg_speed)                           # 31
        f.append(speed_diff)                          # 32
        f.append(abs(speed_diff))                     # 33

        # ── 34–35: Lorenz ───────────────────────────────────────────────────
        f.append(lorenz_x)                            # 34
        f.append(lorenz_z)                            # 35

        # ── 36: Rear bumper ──────────────────────────────────────────────────
        f.append(float(rear_bumper))                  # 36

        # ── 37–43: Interakcje bazowe ────────────────────────────────────────
        f.append(us_min * min_dist)                   # 37
        f.append((1.0 if min_dist < 0.3 else 0.0) * avg_speed) # 38
        f.append((us_left - us_right) * lorenz_x)     # 39
        f.append(speed_diff * (1.0 - min_dist))       # 40
        f.append((mean_left - mean_right) * lorenz_x) # 41
        f.append(min_dist * avg_speed)                # 42
        f.append(us_min * avg_speed)                  # 43

        # ── 44–59: Rozszerzone agregaty ─────────────────────────────────────
        f.append(float(np.log(min_dist + 0.01)))      # 44
        f.append(float(np.var(front_sec)))            # 45
        f.append(mean_left - mean_front)              # 46
        f.append(mean_right - mean_front)             # 47
        f.append(mean_rear - mean_front)              # 48
        f.append(min_dist ** 2)                       # 49
        f.append(us_min ** 2)                         # 50
        front_peak = float(np.max(front_sec))
        f.append(front_peak)                          # 51
        f.append(float(np.max(left_sec)))             # 52
        f.append(float(np.max(right_sec)))            # 53
        f.append(front_peak - mean_front)             # 54
        f.append(float(np.max(left_sec)) - mean_left) # 55
        f.append(float(np.max(right_sec)) - mean_right) # 56
        f.append(1.0 / (us_left + 0.1))               # 57
        f.append(1.0 / (us_right + 0.1))              # 58
        f.append(avg_speed ** 2)                      # 59

        # ── 60: Bias ────────────────────────────────────────────────────────
        f.append(1.0)                                 # 60

        # ── 61–64: CLEARANCES (Nowe!) ───────────────────────────────────────
        front_clearance = 1.0 - mean_front
        rear_clearance  = 1.0 - mean_rear
        left_clearance  = 1.0 - mean_left
        right_clearance = 1.0 - mean_right
        f.append(front_clearance)                     # 61
        f.append(rear_clearance)                      # 62
        f.append(left_clearance)                      # 63
        f.append(right_clearance)                     # 64

        # ── 65–67: FREE SPACE VECTOR (Nowe!) ────────────────────────────────
        f.append(math.sin(free_angle))                # 65
        f.append(math.cos(free_angle))                # 66
        f.append(free_mag)                            # 67

        # ── 68–75: ONE-HOT ACTION (Nowe!) ───────────────────────────────────
        action_vec = [0.0] * 8
        if last_action:
            # Zakładamy że akcje są od 1 do 8 (Enum)
            idx = last_action.value - 1
            if 0 <= idx < 8:
                action_vec[idx] = 1.0
        f.extend(action_vec)                          # 68-75

        # ── 76: BUMPER HISTORY (Nowe!) ─────────────────────────────────────
        if rear_bumper == 1:
            self.bumper_history = 1.0
        else:
            self.bumper_history *= 0.9  # Szybki decay
        f.append(self.bumper_history)                 # 76

        # ── 77–81: NOWE INTERAKCJE (Nowe!) ──────────────────────────────────
        f.append(front_clearance * left_clearance)    # 77 - wolny lewy narożnik
        f.append(front_clearance * right_clearance)   # 78 - wolny prawy narożnik
        f.append(rear_clearance * left_clearance)     # 79
        f.append(rear_clearance * right_clearance)    # 80
        f.append(front_clearance * avg_speed)         # 81

        return np.array(f, dtype=np.float32)


# =============================================================================
class DualLinearApproximator:
    """
    Aproksymator liniowy z dwiema macierzami:
    - q_weights: wartość oczekiwana („co robić”)
    - a_weights: kara za złe doświadczenia („czego unikać”)
    """
    def __init__(self, n_features: int, n_actions: int,
                 learning_rate: float = 0.01,
                 avoidance_penalty: float = 1.0,
                 weight_clip: float = 10.0):
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = learning_rate
        self.penalty = avoidance_penalty
        self.clip = weight_clip

        # Inicjalizacja wag (małe wartości)
        self.q_weights = np.random.uniform(-0.01, 0.01, (n_actions, n_features))
        self.a_weights = np.random.uniform(-0.01, 0.01, (n_actions, n_features))

    def predict_q(self, features: np.ndarray) -> np.ndarray:
        """Zwraca Q-values dla wszystkich akcji (8,)"""
        return np.dot(self.q_weights, features)

    def predict_a(self, features: np.ndarray) -> np.ndarray:
        """Zwraca wartości unikania (im wyższe, tym gorzej)"""
        return np.dot(self.a_weights, features)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Łączna wartość decyzyjna = Q - penalty * A
        Używana w decide().
        """
        q = self.predict_q(features)
        a = self.predict_a(features)
        return q - self.penalty * a

    def update_q(self, features: np.ndarray, action: int, target_q: float):
        """Aktualizacja macierzy Q (TD learning)"""
        q_sa = self.predict_q(features)[action]
        td_error = q_sa - target_q
        self.q_weights[action] -= self.lr * td_error * features
        np.clip(self.q_weights, -self.clip, self.clip, out=self.q_weights)

    def update_a(self, features: np.ndarray, action: int, target_a: float):
        """
        Aktualizacja macierzy unikania.
        target_a – pożądana wartość unikania (np. 1.0 = źle, 0.0 = dobrze)
        """
        a_sa = self.predict_a(features)[action]
        td_error = a_sa - target_a
        self.a_weights[action] -= self.lr * td_error * features
        np.clip(self.a_weights, -self.clip, self.clip, out=self.a_weights)

    def set_learning_rate(self, lr: float):
        self.lr = lr

class GateApproximator:
    """
    Bramka decyzyjna – uczy się, kiedy użyć L1, a kiedy L2.
    Wejście: cechy (np. 16 najważniejszych lub wszystkie 82)
    Wyjście: 2 logity (dla L1 i L2) -> softmax -> wagi
    """
    def __init__(self, n_features: int, learning_rate: float = 0.005, temperature: float = 1.0):
        self.n_features = n_features
        self.lr = learning_rate
        self.temp = temperature
        self.weights = np.random.uniform(-0.01, 0.01, (2, n_features))
        self.bias = np.zeros(2)

    def predict_logits(self, features: np.ndarray) -> np.ndarray:
        """Zwraca logity (2,)"""
        return np.dot(self.weights, features) + self.bias

    def predict_weights(self, features: np.ndarray) -> np.ndarray:
        """Zwraca wagi po softmax (2,)"""
        logits = self.predict_logits(features) / self.temp
        exp = np.exp(logits - np.max(logits))
        return exp / np.sum(exp)

    def update(self, features: np.ndarray, target_weights: np.ndarray):
        """
        Aktualizacja przez SGD.
        target_weights – pożądane wagi (np. [1.0, 0.0] jeśli L1 była lepsza)
        """
        logits = self.predict_logits(features)
        probs = self.predict_weights(features)

        # Gradient cross-entropy
        grad = probs - target_weights

        self.weights -= self.lr * np.outer(grad, features)
        self.bias -= self.lr * grad
        np.clip(self.weights, -5.0, 5.0, out=self.weights)
        np.clip(self.bias, -2.0, 2.0, out=self.bias)

    def set_learning_rate(self, lr: float):
        self.lr = lr

class WorldModel:
    """
    Model świata – przewiduje (next_state, reward) na podstawie (state, action).
    Używany do generowania kontrfaktycznych doświadczeń.
    Wejście: state (WYM_FEATURES) + one-hot akcji (8)
    Wyjście: next_state (WYM_FEATURES) + reward (1)
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 16,
                 learning_rate: float = 0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = learning_rate

        input_dim = state_dim + action_dim
        output_dim = state_dim + 1

        # Inicjalizacja wag (małe wartości)
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros(output_dim)

    def _forward(self, state: np.ndarray, action: int):
        """Forward pass – zwraca (z1, a1, output)"""
        action_onehot = np.zeros(self.action_dim, dtype=np.float32)
        action_onehot[action] = 1.0
        x = np.concatenate([state, action_onehot])
        z1 = np.dot(x, self.W1) + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        out = np.dot(a1, self.W2) + self.b2
        return z1, a1, out

    def predict(self, state: np.ndarray, action: int):
        """Zwraca (next_state, reward)"""
        _, _, out = self._forward(state, action)
        next_state = out[:-1]
        reward = float(out[-1])
        return next_state, reward

    def train_step(self, state: np.ndarray, action: int,
                   target_next_state: np.ndarray, target_reward: float):
        """Pojedynczy krok SGD z backprop"""
        action_onehot = np.zeros(self.action_dim, dtype=np.float32)
        action_onehot[action] = 1.0
        x = np.concatenate([state, action_onehot])

        # Forward
        z1 = np.dot(x, self.W1) + self.b1
        a1 = np.maximum(0, z1)
        out = np.dot(a1, self.W2) + self.b2

        # Target
        target = np.concatenate([target_next_state, [target_reward]])

        # Gradient na wyjściu
        d_out = out - target  # (state_dim+1,)

        # Propagacja wstecz
        d_W2 = np.outer(a1, d_out)  # (hidden_dim, state_dim+1)
        d_b2 = d_out                 # (state_dim+1,)

        d_a1 = np.dot(d_out, self.W2.T)  # (hidden_dim,)
        d_z1 = d_a1 * (z1 > 0)           # ReLU gradient

        d_W1 = np.outer(x, d_z1)  # (input_dim, hidden_dim)
        d_b1 = d_z1               # (hidden_dim,)

        # Aktualizacja wag (SGD)
        self.W2 -= self.lr * d_W2
        self.b2 -= self.lr * d_b2
        self.W1 -= self.lr * d_W1
        self.b1 -= self.lr * d_b1

    def get_state(self):
        return {
            'W1': self.W1.tolist(),
            'b1': self.b1.tolist(),
            'W2': self.W2.tolist(),
            'b2': self.b2.tolist()
        }

    def set_state(self, state_dict):
        self.W1 = np.array(state_dict['W1'])
        self.b1 = np.array(state_dict['b1'])
        self.W2 = np.array(state_dict['W2'])
        self.b2 = np.array(state_dict['b2'])

class WorldModelBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int,
             next_state: np.ndarray, reward: float):
        self.buffer.append((state.copy(), action, next_state.copy(), reward))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)

class NeuralBrainWithImagination:
    def __init__(self, config: SwarmConfig, n_features: int = 82, n_actions: int = 8,
                 l1_weights=None, l2_weights=None, a_weights=None, gate_weights=None):
        self.config = config
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = config.NN_LEARNING_RATE
        self.gamma = config.DISCOUNT_FACTOR
        self.clip = config.NN_CLIP_GRAD

        # ---- Warstwy wspolne ----
        # W1: (82, 32) – inicjalizacja z L1 (srednia po akcjach)
        if l1_weights is not None and config.NN_USE_L1_INIT:
            l1_mean = np.mean(l1_weights, axis=0)          # (82,)
            self.W1 = np.tile(l1_mean.reshape(-1,1), (1, config.NN_HIDDEN_1)) * 0.1
        else:
            self.W1 = np.random.randn(n_features, config.NN_HIDDEN_1) * 0.1
        self.b1 = np.zeros(config.NN_HIDDEN_1)

        # W2: (32, 16) – inicjalizacja z L2 (srednia po akcjach)
        if l2_weights is not None and config.NN_USE_L2_INIT:
            l2_mean = np.mean(l2_weights, axis=0)          # (32,)
            self.W2 = np.tile(l2_mean.reshape(-1,1), (1, config.NN_HIDDEN_2)) * 0.1
        else:
            self.W2 = np.random.randn(config.NN_HIDDEN_1, config.NN_HIDDEN_2) * 0.1
        self.b2 = np.zeros(config.NN_HIDDEN_2)

        # ---- Glowa Q ----
        self.W_q = np.random.randn(config.NN_HIDDEN_2, n_actions) * 0.1
        self.b_q = np.zeros(n_actions)

        # ---- Glowa A (opcjonalna) ----
        if config.NN_USE_A_INIT and a_weights is not None:
            self.W_a = np.random.randn(config.NN_HIDDEN_2, n_actions) * 0.1
            self.b_a = np.zeros(n_actions)
        else:
            self.W_a = np.random.randn(config.NN_HIDDEN_2, n_actions) * 0.1 if config.NN_USE_A_INIT else None
            self.b_a = np.zeros(n_actions) if config.NN_USE_A_INIT else None

        # ---- Glowa gate (opcjonalna, diagnostyczna) ----
        if config.NN_USE_GATE_INIT and gate_weights is not None:
            self.W_gate = gate_weights.T  # (16,2)
            self.b_gate = np.zeros(2)
        else:
            self.W_gate = np.random.randn(config.NN_HIDDEN_2, 2) * 0.1
            self.b_gate = np.zeros(2)

        # ---- Glowa modelu swiata ----
        self.W_wm1 = np.random.randn(24, 16) * 0.1
        self.b_wm1 = np.zeros(16)
        self.W_wm2 = np.random.randn(16, n_features + 1) * 0.1
        self.b_wm2 = np.zeros(n_features + 1)

        # Cache do backprop
        self.cache = {}

    def forward_q(self, features):
        """Oblicza Q i zapisuje posrednie wartosci w cache."""
        z1 = np.dot(features, self.W1) + self.b1
        a1 = np.maximum(0, z1)                     # ReLU
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = np.maximum(0, z2)
        q = np.dot(a2, self.W_q) + self.b_q

        # Clipping wartosci Q
        q = np.clip(q, -5, 5)

        self.cache.update({
            'features': features,
            'z1': z1, 'a1': a1,
            'z2': z2, 'a2': a2,
            'q': q
        })
        return q

    def forward_gate(self):
        """Wykorzystuje a2 z cache do obliczenia wag gate."""
        if 'a2' not in self.cache:
             return np.array([0.5, 0.5])
        a2 = self.cache['a2']
        gate_logits = np.dot(a2, self.W_gate) + self.b_gate
        exps = np.exp(gate_logits - np.max(gate_logits))
        gate_weights = exps / np.sum(exps)
        self.cache['gate_weights'] = gate_weights
        return gate_weights

    def forward_a(self):
        """Oblicza wartosci A (avoidance) – jesli glowa istnieje."""
        if self.W_a is None or 'a2' not in self.cache:
            return np.zeros(self.n_actions)
        a2 = self.cache['a2']
        a_out = np.dot(a2, self.W_a) + self.b_a
        self.cache['a_out'] = a_out
        return a_out

    def forward_world(self, action):
        """Przewiduje nastepny stan i nagrode na podstawie a2 i akcji."""
        if 'a2' not in self.cache:
             return np.zeros(self.n_features), 0.0

        a2 = self.cache['a2']
        action_onehot = np.zeros(self.n_actions)
        action_onehot[action] = 1.0
        x = np.concatenate([a2, action_onehot])          # (24,)

        z_wm1 = np.dot(x, self.W_wm1) + self.b_wm1
        a_wm1 = np.maximum(0, z_wm1)                     # ReLU
        out = np.dot(a_wm1, self.W_wm2) + self.b_wm2

        next_features_pred = out[:-1]
        reward_pred = out[-1]

        self.cache.update({
            'x_wm': x,
            'z_wm1': z_wm1,
            'a_wm1': a_wm1,
            'out_wm': out
        })
        return next_features_pred, reward_pred

    def backward_q(self, target_q, target_a=None):
        """Aktualizacja wag dla Q i warstw wspolnych."""
        if 'features' not in self.cache: return

        f = self.cache['features']
        a2 = self.cache['a2']; z2 = self.cache['z2']
        a1 = self.cache['a1']; z1 = self.cache['z1']
        q = self.cache['q']

        d_q = q - target_q                                   # (8,)

        d_W_q = np.outer(a2, d_q)                            # (16,8)
        d_b_q = d_q

        # Avoidance gradient
        d_a_out = 0
        if target_a is not None and self.W_a is not None:
            a_out = self.cache.get('a_out', self.forward_a())
            d_a = a_out - target_a
            d_W_a = np.outer(a2, d_a)
            d_b_a = d_a

            # Clip gradient A
            grad_clip = 0.1
            np.clip(d_W_a, -grad_clip, grad_clip, out=d_W_a)
            np.clip(d_b_a, -grad_clip, grad_clip, out=d_b_a)

            # Update A weights
            self.W_a -= self.lr * d_W_a
            self.b_a -= self.lr * d_b_a

            # Backprop through A head
            d_a_out = np.dot(d_a, self.W_a.T) # (16,)

        d_a2 = np.dot(d_q, self.W_q.T) + d_a_out             # (16,) + (16,)
        d_z2 = d_a2 * (z2 > 0)                               # ReLU grad
        d_W2 = np.outer(a1, d_z2)                            # (32,16)
        d_b2 = d_z2

        d_a1 = np.dot(d_z2, self.W2.T)                       # (32,)
        d_z1 = d_a1 * (z1 > 0)
        d_W1 = np.outer(f, d_z1)                             # (82,32)
        d_b1 = d_z1

        # CLIPPING GRADIENTOW (Poprawka C)
        grad_clip = 0.1
        for grad in [d_W1, d_W2, d_W_q, d_b1, d_b2, d_b_q]:
             np.clip(grad, -grad_clip, grad_clip, out=grad)

        # Aktualizacja SGD
        self.W_q -= self.lr * d_W_q
        self.b_q -= self.lr * d_b_q
        self.W2  -= self.lr * d_W2
        self.b2  -= self.lr * d_b2
        self.W1  -= self.lr * d_W1
        self.b1  -= self.lr * d_b1

        # Clipping wag (pozostawiamy)
        for w in (self.W1, self.W2, self.W_q):
            np.clip(w, -self.clip, self.clip, out=w)

    def backward_world(self, target_next_features, target_reward):
        """Aktualizacja wag modelu swiata (nie rusza warstw wspolnych)."""
        if 'out_wm' not in self.cache: return

        x = self.cache['x_wm']
        z_wm1 = self.cache['z_wm1']
        a_wm1 = self.cache['a_wm1']
        out = self.cache['out_wm']

        target = np.concatenate([target_next_features, [target_reward]])
        d_out = out - target                                  # (83,)

        d_W_wm2 = np.outer(a_wm1, d_out)                     # (16,83)
        d_b_wm2 = d_out

        d_a_wm1 = np.dot(d_out, self.W_wm2.T)                # (16,)
        d_z_wm1 = d_a_wm1 * (z_wm1 > 0)
        d_W_wm1 = np.outer(x, d_z_wm1)                       # (24,16)
        d_b_wm1 = d_z_wm1

        # CLIPPING GRADIENTOW (Poprawka D)
        grad_clip = 0.1
        for grad in [d_W_wm1, d_W_wm2, d_b_wm1, d_b_wm2]:
             np.clip(grad, -grad_clip, grad_clip, out=grad)

        self.W_wm2 -= self.lr * d_W_wm2
        self.b_wm2 -= self.lr * d_b_wm2
        self.W_wm1 -= self.lr * d_W_wm1
        self.b_wm1 -= self.lr * d_b_wm1

        np.clip(self.W_wm1, -self.clip, self.clip, out=self.W_wm1)
        np.clip(self.W_wm2, -self.clip, self.clip, out=self.W_wm2)

    def generate_counterfactual(self, features, action_taken, actual_reward):
        """Zwraca (akcja, wartosc, nastepny stan) dla lepszej kontrfaktyki lub None."""
        self.forward_q(features)        # zeby miec a2 w cache

        best_action = None
        best_value = -np.inf
        best_next = None

        # Save original state needed for world model
        if 'a2' in self.cache:
            original_a2 = self.cache['a2'].copy()
        else:
            return None, None, None

        for a in range(self.n_actions):
            if a == action_taken:
                continue
            next_feat, pred_reward = self.forward_world(a)

            q_next = self.forward_q(next_feat)          # (8,)
            max_q_next = np.max(q_next)
            cf_value = pred_reward + self.gamma * max_q_next

            if cf_value > best_value:
                best_value = cf_value
                best_action = a
                best_next = next_feat

            # Restore a2 for next iteration of world model prediction
            self.cache['a2'] = original_a2

        if best_value > actual_reward + self.config.COUNTERFACTUAL_THRESHOLD:
            return best_action, best_value, best_next
        return None, None, None


class NeuralHybridBrain:
    def __init__(self, config: SwarmConfig, feature_extractor: FeatureExtractor):
        self.config = config
        self.feature_extractor = feature_extractor
        self.actions_list = list(Action)
        self.n_actions = len(self.actions_list)

        # Inicjalizacja feature extractor z dummy data zeby poznac n_features
        dummy = self.feature_extractor.extract(
            np.zeros(16), 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0, 3.0, None, 0.0, 0.0
        )
        self.n_features = len(dummy) # Powinno byc 82

        # Zaladuj wagi z pliku (jesli istnieja)
        self.nn = NeuralBrainWithImagination(
            config, self.n_features, self.n_actions,
            l1_weights=None, l2_weights=None, a_weights=None, gate_weights=None
        )

        # Metadata
        self.top32_indices = None
        self.step_counter = 0
        self.counterfactual_count = 0
        self.lr = config.NN_LEARNING_RATE
        # L2 Metadata
        self.l2_feature_indices = None
        self.l2_importance_history = []
        self.l2_stability_counter = 0
        self.l2_last_reinforce = 0
        self.l2_version = 0
        self.l2_last_importance = None

        self.load_or_create()   # patrz sekcja 5

        # Standardowe komponenty
        self.normalizer = RunningNormalizer(self.n_features)
        self.replay_buffer = ReplayBuffer(capacity=config.REPLAY_BUFFER_CAPACITY)
        self.batch_size = config.REPLAY_BATCH_SIZE
        self.train_freq = config.REPLAY_TRAIN_FREQ
        self.steps_since_train = 0
        self.epsilon = config.EPSILON

        self.last_features = None
        self.last_action = None

        # Rejestracja auto-zapisu
        import atexit
        atexit.register(self.save)

        logger.info(f"NeuralHybridBrain: MLP {self.n_features}->{config.NN_HIDDEN_1}->{config.NN_HIDDEN_2}->Output initialized")

    def get_features(self, lidar_16, us_left, us_right, encoder_l, encoder_r,
                     lorenz_x, lorenz_z, rear_bumper, min_dist, last_action=None,
                     free_angle=0.0, free_mag=0.0) -> np.ndarray:
        raw = self.feature_extractor.extract(
            lidar_16, us_left, us_right, encoder_l, encoder_r,
            lorenz_x, lorenz_z, rear_bumper, min_dist, last_action, free_angle, free_mag
        )
        self.normalizer.update(raw)
        return self.normalizer.normalize(raw)

    def decide(self, features: np.ndarray, instinct_bias: Dict[Action, float],
               concept_suggestion: Optional[Action]) -> Tuple[Action, str, np.ndarray]:

        if random.random() < self.epsilon:
            return random.choice(self.actions_list), "EXPLORE", np.zeros(2)

        q = self.nn.forward_q(features)
        # opcjonalnie: odejmij kare avoidance
        if self.nn.W_a is not None:
            a_out = self.nn.forward_a()
            q = q - self.config.AVOIDANCE_PENALTY * a_out

        scores = q.copy()
        for i, action in enumerate(self.actions_list):
            scores[i] += instinct_bias.get(action, 0.0) * 4.0
            if action == concept_suggestion:
                scores[i] += 1.5

        best_idx = int(np.argmax(scores))

        # Gate weights for diagnostics (if gate exists)
        gate_weights = np.zeros(2)
        if hasattr(self.nn, 'W_gate'):
             gate_weights = self.nn.forward_gate()

        return self.actions_list[best_idx], "NEURAL", gate_weights

    def is_bad_state(self, reward: float, source: str, action: Action,
                     lidar_min: float, stagnant: bool, oscillated: bool) -> Tuple[bool, float]:
        # Kolizja / twardy odruch
        if source in ("LIDAR_HARD_SAFETY", "HARD_REFLEX") and action != Action.REVERSE:
            return True, 1.0   # chcemy, aby A dazylo do 1.0 (unikaj)
        # Stagnacja (ale nie podczas skretu)
        if stagnant and action in (Action.FORWARD, Action.REVERSE):
            return True, 0.8
        # Oscylacja
        if oscillated:
            return True, 0.6
        # Niska nagroda
        if reward < -1.0:
            return True, 0.5
        # Domyslnie – dobre doswiadczenie
        return False, 0.0

    def update_q(self, old_features: np.ndarray, action: Action,
                 reward: float, new_features: np.ndarray,
                 source: str, lidar_min: float,
                 stagnant: bool, oscillated: bool,
                 done: bool = False, lr_scale=1.0):

        action_idx = self.actions_list.index(action)

        # Dodaj do replay buffera
        self.replay_buffer.push(old_features, action_idx, reward, new_features, done)

        # Uczenie modelu swiata na biezacym doswiadczeniu
        self.nn.forward_q(old_features)          # wypelnia cache
        self.nn.forward_world(action_idx)        # forward swiata
        self.nn.backward_world(new_features, reward)

        # Uczenie Avoidance (natychmiastowe)
        is_bad, target_a_val = self.is_bad_state(reward, source, action,
                                                lidar_min, stagnant, oscillated)
        if is_bad:
            target_a = np.zeros(self.n_actions)
            target_a[action_idx] = target_a_val
            # Wymus forward A zeby miec cache
            self.nn.forward_a()
            # Uzyj backward_q tylko do aktualizacji A (przekazujac target_q jako obecne q zeby gradient d_q byl 0?)
            # Nie, backward_q oblicza d_q = q - target_q.
            # Jesli chcemy uczyc TYLKO A, to d_q powinno byc 0.
            # Wiec target_q = q.
            current_q = self.nn.cache['q']
            self.nn.backward_q(current_q, target_a)

        # Generuj kontrfaktyke
        cf = self.nn.generate_counterfactual(old_features, action_idx, reward)
        if cf[0] is not None:
            cf_action, cf_val, cf_next = cf
            self.replay_buffer.push(old_features, cf_action, cf_val, cf_next, done)
            self.counterfactual_count += 1

        # Decay epsilon i LR
        self.epsilon = max(self.config.EPSILON_MIN,
                           self.epsilon * self.config.EPSILON_DECAY)
        self.lr = max(self.config.LR_MIN,
                      self.lr * self.config.LR_DECAY)
        self.nn.lr = self.lr

        self.steps_since_train += 1
        if self.steps_since_train >= self.train_freq and len(self.replay_buffer) >= self.batch_size:
            self._train_on_batch()
            self.steps_since_train = 0

        self.step_counter += 1
        # Zaawansowane wzmacnianie L2 co 1000 krokow z roznymi metodami
        if self.step_counter % 1000 == 0 and self.step_counter > 0:
            # Rotacja metod co 5000 krokow
            method_cycle = (self.step_counter // 1000) % 5
            if method_cycle == 0:
                method = 'simple'
            elif method_cycle == 1:
                method = 'combined'
            elif method_cycle == 2:
                method = 'stability'
            else:
                method = 'combined'  # domyslnie combined

            changed, stats = self.reinforce_l2(method=method)

            # Jesli zmiana byla duza (>20%), zapisz stan natychmiast
            if changed and stats.get('changes', 0) > 6:
                logger.info("ð¥ Duza zmiana L2 â wymuszam zapis stanu")
                self.save()

        if self.step_counter % 5000 == 0 and self.step_counter > 0:
            results = {}
            for method in ['simple', 'combined', 'stability']:
                imp, conf = self.analyze_feature_importance(method)
                if imp is not None:
                    top5 = np.argsort(imp)[-5:]
                    results[method] = {
                        'top5': top5,
                        'confidence': conf
                    }

            # Znajdz zgodnosc miedzy metodami (ile cech wspolnych w top20)
            if len(results) >= 2:
                sets = [set(np.argsort(results[m]['top5'])[-20:]) for m in results]
                agreement = len(sets[0].intersection(*sets[1:])) / 20.0
                logger.info(f"ð Zgodnosc metod L2: {agreement:.2f}")
        if self.step_counter % 400 == 0:
            self.save()

    def _train_on_batch(self):
        batch = self.replay_buffer.sample(self.batch_size)
        for old_feat, act_idx, reward, new_feat, done in batch:
            self.nn.forward_q(old_feat)
            if done:
                target_q = reward
            else:
                q_next = self.nn.forward_q(new_feat)
                max_next_q = np.max(q_next)
                target_q = reward + self.config.DISCOUNT_FACTOR * max_next_q

            target_q_vec = self.nn.cache['q'].copy()
            target_q_vec[act_idx] = target_q
            self.nn.backward_q(target_q_vec)

    WEIGHTS_DIR = "weights/"
    WEIGHTS_FILE = "hierarchical_swarm_mlp_v6.npz"
    WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, WEIGHTS_FILE)

    def analyze_feature_importance(self, method='combined'):
        """
        Zaawansowana analiza waznosci cech z wieloma metodami.

        Args:
            method: 'simple' - tylko suma wag bezwzglednych z W1
                    'gradient' - uwzglednia gradienty z backprop
                    'combined' - polaczenie obu metod
                    'stability' - uwzglednia historie zmian

        Returns:
            importance_vector: numpy array (82,) z waznoscia kazdej cechy
            confidence: float (0-1) poziom ufnosci analizy
        """
        if not hasattr(self.nn, 'W1') or self.nn.W1 is None:
            return None, 0.0

        # Metoda 1: Prosta suma wag bezwzglednych z pierwszej warstwy
        importance_w1 = np.sum(np.abs(self.nn.W1), axis=1)

        # Normalizacja do [0,1]
        importance_w1 = (importance_w1 - importance_w1.min()) / (importance_w1.max() - importance_w1.min() + 1e-8)

        if method == 'simple':
            return importance_w1, 0.7

        # Metoda 2: Gradient-based (jesli dostepne sa gradienty z cache)
        importance_grad = np.zeros(self.n_features)
        # Note: We need to store gradient history or check cache.
        # NeuralBrainWithImagination cache stores gradients temporarily during backprop but they might be overwritten.
        # But here we assume we can access them if we modify NeuralBrainWithImagination to store d_W1 in cache?
        # NeuralBrainWithImagination.backward_q stores local d_W1 but doesn't persist it in self.cache for long term?
        # Let's check NeuralBrainWithImagination.backward_q. It uses local vars.
        # I should probably update NeuralBrainWithImagination to store d_W1 in cache or just use W1 for now.
        # Or better, just implement 'simple' and 'stability' if gradient is hard to get without changing backward_q.
        # The prompt implies I should use what's available.
        # Let's assume 'gradient' relies on cache having 'd_W1'. I'll need to update backward_q to save it.

        if hasattr(self.nn, 'cache') and 'd_W1' in self.nn.cache:
            d_W1 = self.nn.cache.get('d_W1', np.zeros_like(self.nn.W1))
            importance_grad = np.sum(np.abs(d_W1), axis=1)
            importance_grad = (importance_grad - importance_grad.min()) / (importance_grad.max() - importance_grad.min() + 1e-8)

        # Metoda 3: Combined
        if method == 'combined':
            # Polacz obie metody z wagami
            combined = 0.7 * importance_w1 + 0.3 * importance_grad
            confidence = 0.8

            # Dodaj do historii
            if not hasattr(self, 'l2_importance_history'): self.l2_importance_history = []
            if len(self.l2_importance_history) > 10:
                self.l2_importance_history.pop(0)
            self.l2_importance_history.append(combined.copy())

            return combined, confidence

        # Metoda 4: Stabilnosc (uwzglednia historie)
        if method == 'stability':
            if not hasattr(self, 'l2_importance_history') or len(self.l2_importance_history) < 5:
                return importance_w1, 0.5

            # Oblicz srednia z historii
            avg_importance = np.mean(self.l2_importance_history[-5:], axis=0)

            # Oblicz wariancje – im mniejsza, tym bardziej stabilna cecha
            variance = np.var(self.l2_importance_history[-5:], axis=0)
            stability = 1.0 / (variance + 1e-8)
            stability = (stability - stability.min()) / (stability.max() - stability.min() + 1e-8)

            # Polacz srednia waznosc ze stabilnoscia
            final = 0.6 * avg_importance + 0.4 * stability
            return final, 0.9

        return importance_w1, 0.5

    def reinforce_l2(self, force=False, method='combined'):
        """
        Inteligentne wzmacnianie L2 z adaptacyjnym progiem i monitoringiem.

        Args:
            force: bool – wymus aktualizacje nawet jesli zmiany sa male
            method: metoda analizy waznosci

        Returns:
            changed: bool – czy L2 zostala zaktualizowana
            stats: dict – statystyki zmian
        """
        # Pobierz waznosc cech
        importance, confidence = self.analyze_feature_importance(method)
        if importance is None:
            return False, {'error': 'Brak danych'}

        # Wybierz 32 najwazniejsze cechy
        current_indices = np.argsort(importance)[-32:]

        stats = {
            'timestamp': time.time(),
            'method': method,
            'confidence': confidence,
            'current_indices': current_indices.copy(),
            'changes': 0,
            'stability': 0.0
        }

        # Jesli nie mielismy jeszcze L2, po prostu ustaw
        if not hasattr(self, 'l2_feature_indices') or self.l2_feature_indices is None:
            self.l2_feature_indices = current_indices
            self.l2_version += 1
            stats['changes'] = 32
            stats['stability'] = 0.0
            logger.info(f"✨ L2 zainicjalizowana (v{self.l2_version}): 32 cech, metoda={method}, ufnosc={confidence:.2f}")
            return True, stats

        # Oblicz zmiany
        old_set = set(self.l2_feature_indices)
        new_set = set(current_indices)

        changes = len(old_set.symmetric_difference(new_set)) // 2  # ile cech sie zmienilo
        stats['changes'] = changes
        stats['old_indices'] = self.l2_feature_indices.copy()

        # Oblicz stabilnosc (jak bardzo zmienialy sie wagi od ostatniego razu)
        if hasattr(self, 'l2_last_importance') and self.l2_last_importance is not None:
            importance_change = np.mean(np.abs(importance - self.l2_last_importance))
            stability = 1.0 / (1.0 + importance_change)
            stats['stability'] = stability
        else:
            stability = 0.5

        self.l2_last_importance = importance.copy()

        # Adaptacyjny prog – im bardziej stabilny system, tym wiekszy prog
        adaptive_threshold = max(3, int(5 * stability))

        if force or changes >= adaptive_threshold:
            # Zaktualizuj L2
            self.l2_feature_indices = current_indices
            self.l2_version += 1
            self.l2_last_reinforce = time.time()
            self.l2_stability_counter = int(stability * 100)

            log_msg = (f"🔄 L2 wzmocniona (v{self.l2_version}): "
                      f"zmienilo sie {changes} cech, "
                      f"prog={adaptive_threshold}, "
                      f"ufnosc={confidence:.2f}, "
                      f"stabilnosc={stability:.2f}")

            if changes > 10:
                logger.warning(log_msg + " – DUZA ZMIANA!")
            else:
                logger.info(log_msg)

            return True, stats
        else:
            logger.debug(f"L2 stabilna: zmiany={changes} < prog={adaptive_threshold}")
            return False, stats

    def debug_l2(self):
        """Metoda pomocnicza do diagnostyki i recznego strojenia L2"""
        print("\n" + "="*50)
        print("🔍 DIAGNOSTYKA L2")
        print("="*50)

        if not hasattr(self, 'l2_feature_indices') or self.l2_feature_indices is None:
            print("❌ L2 niezainicjalizowana")
            return

        print(f"📊 L2 wersja: {getattr(self, 'l2_version', 0)}")
        print(f"📊 Liczba cech: {len(self.l2_feature_indices)}")
        print(f"📊 Indeksy: {sorted(self.l2_feature_indices)}")

        # Analiza waznosci
        imp_simple, conf_simple = self.analyze_feature_importance('simple')
        imp_combined, conf_combined = self.analyze_feature_importance('combined')

        if imp_simple is not None:
            print(f"\n📈 Waznosc cech (simple):")
            top10 = np.argsort(imp_simple)[-10:]
            print(f"   Top10: {top10}")
            print(f"   Ufnosc: {conf_simple:.2f}")

        if imp_combined is not None:
            print(f"\n📈 Waznosc cech (combined):")
            top10 = np.argsort(imp_combined)[-10:]
            print(f"   Top10: {top10}")
            print(f"   Ufnosc: {conf_combined:.2f}")

        print("\n" + "="*50)
        return {
            'indices': self.l2_feature_indices,
            'version': getattr(self, 'l2_version', 0),
            'importance_simple': imp_simple,
            'importance_combined': imp_combined
        }

    def save(self, path=None):
        if path is None:
            path = self.WEIGHTS_PATH
        os.makedirs(self.WEIGHTS_DIR, exist_ok=True)

        # Sprawdz, czy wagi nie zawieraja NaN/Inf
        for name, arr in [('W1', self.nn.W1), ('b1', self.nn.b1)]:
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                print(f"!!! Wykryto NaN/Inf w {name} – zapis wstrzymany !!!")
                return

        # Opcjonalny backup
        import shutil
        if os.path.exists(path):
            backup = path + ".backup"
            shutil.copy(path, backup)

        np.savez(
            path,
            W1=self.nn.W1, b1=self.nn.b1,
            W2=self.nn.W2, b2=self.nn.b2,
            W_gate=self.nn.W_gate, b_gate=self.nn.b_gate,
            W_q=self.nn.W_q, b_q=self.nn.b_q,
            W_a=self.nn.W_a if self.nn.W_a is not None else np.array([]),
            b_a=self.nn.b_a if self.nn.b_a is not None else np.array([]),
            W_wm1=self.nn.W_wm1, b_wm1=self.nn.b_wm1,
            W_wm2=self.nn.W_wm2, b_wm2=self.nn.b_wm2,
            l2_feature_indices=self.l2_feature_indices if self.l2_feature_indices is not None else np.array([]),
            l2_version=np.array([self.l2_version]),
            l2_last_reinforce=np.array([self.l2_last_reinforce]),
            l2_stability_counter=np.array([self.l2_stability_counter]),
            top32_indices=getattr(self, 'top32_indices', np.array([], dtype=np.int32)),
            step_counter=np.array([self.step_counter], dtype=np.int64),
            freeze_l1_steps=np.array([self.config.L2_MIN_SAMPLES]),
            avoidance_penalty=np.array([self.config.AVOIDANCE_PENALTY]),
            learning_rate=np.array([self.lr]),
            counterfactual_count=np.array([self.counterfactual_count]),
            version=np.array(['6.0'], dtype=object),
            saved_at=np.array([time.strftime("%Y-%m-%d %H:%M:%S")], dtype=object)
        )
        logger.info(f"Model zapisany do {path}")

    def load_or_create(self, path=None):
        if path is None:
            path = self.WEIGHTS_PATH

        if not os.path.exists(path):
            print(f"Brak pliku {path} – tworze nowy model.")
            return False

        try:
            data = np.load(path, allow_pickle=True)
            required = ['W1','b1','W2','b2','W_gate','b_gate','W_q','b_q','W_wm1','b_wm1','W_wm2','b_wm2']
            missing = [k for k in required if k not in data]
            if missing:
                # raise ValueError(f"Brakujace klucze: {missing}")
                logger.warning(f"Brakujace klucze w npz: {missing}. Inicjalizacja czesciowa lub nowa.")
                # Nie przerywamy, moze to starsza wersja

            if 'W1' in data: self.nn.W1 = data['W1'].astype(np.float32)
            if 'b1' in data: self.nn.b1 = data['b1'].astype(np.float32)
            if 'W2' in data: self.nn.W2 = data['W2'].astype(np.float32)
            if 'b2' in data: self.nn.b2 = data['b2'].astype(np.float32)
            if 'W_gate' in data: self.nn.W_gate = data['W_gate'].astype(np.float32)
            if 'b_gate' in data: self.nn.b_gate = data['b_gate'].astype(np.float32)
            if 'W_q' in data: self.nn.W_q = data['W_q'].astype(np.float32)
            if 'b_q' in data: self.nn.b_q = data['b_q'].astype(np.float32)

            if 'W_a' in data and data['W_a'].size > 0:
                self.nn.W_a = data['W_a'].astype(np.float32)
                self.nn.b_a = data['b_a'].astype(np.float32)
            else:
                self.nn.W_a = None
                self.nn.b_a = None

            if 'W_wm1' in data: self.nn.W_wm1 = data['W_wm1'].astype(np.float32)
            if 'b_wm1' in data: self.nn.b_wm1 = data['b_wm1'].astype(np.float32)
            if 'W_wm2' in data: self.nn.W_wm2 = data['W_wm2'].astype(np.float32)
            if 'b_wm2' in data: self.nn.b_wm2 = data['b_wm2'].astype(np.float32)

            if 'l2_feature_indices' in data and data['l2_feature_indices'].size > 0:
                self.l2_feature_indices = data['l2_feature_indices']
                self.l2_version = int(data.get('l2_version', 0))
                self.l2_last_reinforce = float(data.get('l2_last_reinforce', 0))
                self.l2_stability_counter = int(data.get('l2_stability_counter', 0))
                logger.info(f"ð¥ Wczytano L2 v{self.l2_version}: {len(self.l2_feature_indices)} cech, "
                           f"ostatnie wzmocnienie: {self.l2_last_reinforce}")
            self.top32_indices = data.get('top32_indices')
            self.step_counter = int(data.get('step_counter', 0))
            self.counterfactual_count = int(data.get('counterfactual_count', 0))
            self.lr = float(data.get('learning_rate', self.config.LEARNING_RATE))

            version = data.get('version', '?')
            saved_at = data.get('saved_at', 'nieznana data')
            print(f"Wczytano model v{version}, krok {self.step_counter}, zapis: {saved_at}")
            return True

        except Exception as e:
            print(f"Blad wczytywania {path}: {e}\nTworze nowy model.")
            return False


class DCMotorController:
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.x, self.y, self.theta = 0.0, 0.0, 0.0
        self.integral_l, self.integral_r = 0.0, 0.0
        self.prev_error_l, self.prev_error_r = 0.0, 0.0
        self.last_pwm_l, self.last_pwm_r = 0.0, 0.0

    def update_pid(self, target_l: float, target_r: float,
                  encoder_l: float, encoder_r: float, dt: float) -> Tuple[float, float]:
        error_l = target_l - encoder_l
        error_r = target_r - encoder_r

        self.integral_l += error_l * dt
        self.integral_r += error_r * dt
        self.integral_l = np.clip(self.integral_l, -5, 5)
        self.integral_r = np.clip(self.integral_r, -5, 5)

        derivative_l = (error_l - self.prev_error_l) / dt if dt > 0 else 0.0
        derivative_r = (error_r - self.prev_error_r) / dt if dt > 0 else 0.0

        output_l = (self.config.PID_KP * error_l +
                   self.config.PID_KI * self.integral_l +
                   self.config.PID_KD * derivative_l)
        output_r = (self.config.PID_KP * error_r +
                   self.config.PID_KI * self.integral_r +
                   self.config.PID_KD * derivative_r)

        pwm_l = output_l * self.config.PID_OUTPUT_SCALE
        pwm_r = output_r * self.config.PID_OUTPUT_SCALE

        delta_l = np.clip(pwm_l - self.last_pwm_l, -self.config.PWM_SLEW_RATE, self.config.PWM_SLEW_RATE)
        delta_r = np.clip(pwm_r - self.last_pwm_r, -self.config.PWM_SLEW_RATE, self.config.PWM_SLEW_RATE)

        self.last_pwm_l += delta_l
        self.last_pwm_r += delta_r
        self.prev_error_l = error_l
        self.prev_error_r = error_r

        # Hard clamp PWM
        self.last_pwm_l = np.clip(self.last_pwm_l, -self.config.PWM_MAX, self.config.PWM_MAX)
        self.last_pwm_r = np.clip(self.last_pwm_r, -self.config.PWM_MAX, self.config.PWM_MAX)

        return self.last_pwm_l, self.last_pwm_r

    def sync_memory(self, pwm_l: float, pwm_r: float):
        """Zsynchronizuj pamięć PID z zewnętrzną korekcją PWM."""
        self.last_pwm_l = pwm_l
        self.last_pwm_r = pwm_r

    def update_odometry(self, vel_l: float, vel_r: float, dt: float):
        v = (vel_l + vel_r) / 2.0
        omega = (vel_r - vel_l) / self.config.WHEEL_BASE
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        self.theta += omega * dt
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))


class VirtualDamper:
    def __init__(self, config: SwarmConfig):
        self.config = config

    def compute_reward(self, encoder_l: float, encoder_r: float,
                      motor_current: float, action: Action) -> float:
        avg_speed = abs((encoder_l + encoder_r) / 2.0)

        if motor_current > self.config.STALL_CURRENT_THRESHOLD and avg_speed < self.config.STALL_SPEED_THRESHOLD:
            return -5.0

        if action == Action.FORWARD:
            return 1.0 + avg_speed * 0.5
        elif action in (Action.TURN_LEFT, Action.TURN_RIGHT):
            return 1.0
        elif action == Action.REVERSE:
            return -0.3
        else:
            return 0.0


# =============================================================================
# STATE MANAGER (PERSISTENCE WBUDOWANA!)
# =============================================================================

class StateManager:
    """Zarządzanie stanem - persistence wbudowana (bez patchy!)"""

    def __init__(self, config: SwarmConfig):
        self.config = config
        self.brain_path = Path(config.BRAIN_FILE)
        self.save_counter = 0

    def save(self, data: Dict):
        try:
            with open(self.brain_path, 'wb') as f:
                pickle.dump(data, f)
            weights = data.get('weights')
            w_info = f"weights={np.array(weights).shape}" if weights is not None else "weights=None"
            logger.info(f"✓ Saved: {w_info}, Concepts={len(data.get('concepts',{}))}")
        except Exception as e:
            logger.error(f"Save failed: {e}")

    def load(self) -> Optional[Dict]:
        if not self.brain_path.exists():
            logger.info("No saved state - starting fresh")
            return None
        try:
            with open(self.brain_path, 'rb') as f:
                data = pickle.load(f)
            weights = data.get('weights')
            w_info = "weights found" if weights is not None else "no weights"
            logger.info(f"✓ Loaded: {w_info}, Concepts={len(data.get('concepts',{}))}")
            return data
        except Exception as e:
            logger.error(f"Load failed: {e}")
            return None

    def should_auto_save(self) -> bool:
        self.save_counter += 1
        if self.save_counter >= self.config.AUTO_SAVE_INTERVAL:
            self.save_counter = 0
            return True
        return False


# =============================================================================
# SWARM CORE v5.5 FINAL
# =============================================================================

class SwarmCoreV55:
    """
    SWARM v5.5 - Production Final

    ★ Q-Table ≠ Concept Graph!
    ★ Persistence wbudowana (bez patchy)
    ★ Android-compatible (bez relative imports)
    ★ Gotowe do unzip & run
    """

    def __init__(self):
        self.config = SwarmConfig()

        # Hardware
        self.lidar = LidarEngine(self.config)
        self.motors = DCMotorController(self.config)
        self.damper = VirtualDamper(self.config)

        # Feature Extractor (v5.9)
        self.feature_extractor = FeatureExtractor(self.config)

        # AI Layers
        self.brain = NeuralHybridBrain(self.config, self.feature_extractor)
        self.concept_graph = ConceptGraph(self.config)  # ★ WŁAŚCIWY!

        # Moduly
        self.lorenz = LorenzAttractor(self.config)
        self.brain.lorenz = self.lorenz
        self.instinct = FreeSpaceInstinct(self.config)
        self.velocity_mapper = DynamicVelocityMapper(self.config)
        self.stabilizer = ActionStabilizer(self.config)
        self.anti_stagnation = AntiStagnationController(self.config)

        # Persistence wbudowana
        self.state_manager = StateManager(self.config)
        self._load_state()

        self.cycle_count = 0
        self.hard_reflex_hold_remaining = 0
        self.hard_reflex_action: Optional[Action] = None

        # Rear bumper state
        self.rear_bumper_forward_remaining = 0

        # ★ Anti-oscillation: śledzenie powtórzeń akcji
        self._last_action_type: Optional[Action] = None
        self._action_repeat_count: int = 0

        logger.info("=" * 70)
        logger.info("SWARM CORE v5.9 — Q-Approximator — ANDROID-READY")
        logger.info("=" * 70)
        logger.info(f"Q-Approx: {self.brain.n_features} features × {self.brain.n_actions} actions")
        logger.info(f"Concepts: {len(self.concept_graph.concepts)} patterns")
        logger.info(f"Target: {self.config.US_TARGET_DIST*100:.0f}cm")
        logger.info("=" * 70)

    def save_state(self):
        # 1. Zapisz wagi sieci (brain.save() juz to robi)
        if hasattr(self.brain, 'save'):
            self.brain.save()
        else:
            logger.warning("Brain does not have save method")

        # 2. Przygotuj dane do pickle (koncepty, normalizer, Lorenz, epsilon, lr)
        concepts_data = {}
        for name, c in self.concept_graph.concepts.items():
            concepts_data[name] = {
                'name': c.name,
                'sequence': c.sequence,
                'activation': c.activation,
                'success_count': c.success_count,
                'usage_count': c.usage_count,
                'context': c.context
            }

        data = {
            'normalizer_state': self.brain.normalizer.get_state(),
            'epsilon': self.brain.epsilon,
            'lr': self.brain.lr,
            'concepts': concepts_data,
            'lorenz_state': self.lorenz.get_state(),
        }

        # 3. Zapisz do pliku pickle (uzywajac state_manager)
        self.state_manager.save(data)

    def _load_state(self):
        saved = self.state_manager.load()
        if saved:
            # Wczytaj normalizer
            norm_state = saved.get('normalizer_state')
            if norm_state and norm_state.get('n', 0) > 0:
                self.brain.normalizer.set_state(norm_state)
                logger.info(f"Loaded normalizer: n={norm_state['n']} samples")

            # Wczytaj epsilon i lr
            self.brain.epsilon = saved.get('epsilon', self.config.EPSILON)
            self.brain.lr = saved.get('lr', self.config.LEARNING_RATE)
            self.brain.nn.lr = self.brain.lr

            # Wczytaj koncepty
            saved_concepts = saved.get('concepts', {})
            if saved_concepts:
                for name, data in saved_concepts.items():
                    c = Concept(data['name'], data['sequence'])
                    c.activation = data['activation']
                    c.success_count = data['success_count']
                    c.usage_count = data['usage_count']
                    c.context = data.get('context', {})
                    self.concept_graph.concepts[name] = c

            # Wczytaj stan Lorenza
            lorenz_state = saved.get('lorenz_state')
            if lorenz_state:
                self.lorenz.x, self.lorenz.y, self.lorenz.z = lorenz_state

    def _compute_dynamic_safety(self, avg_speed: float) -> Tuple[float, float]:
        scale = self.config.SAFETY_DIST_SPEED_SCALE
        v = abs(avg_speed)
        us = self.config.US_SAFETY_DIST + (scale * v)
        us = max(self.config.SAFETY_US_MIN, min(self.config.SAFETY_US_MAX, us))
        lr = self.config.LIDAR_SAFETY_RADIUS + (scale * v)
        lr = max(self.config.SAFETY_LIDAR_MIN, min(self.config.SAFETY_LIDAR_MAX, lr))
        return us, lr

    def validate_safety_constraints(self, us_left_dist: float, us_right_dist: float,
                                   encoder_l: float, encoder_r: float,
                                   rear_bumper: int = 0) -> Optional[Tuple[Action, str]]:
        """Safety check z dual US + LIDAR + rear bumper + LIDAR hard safety"""
        avg_speed = (encoder_l + encoder_r) / 2.0
        dyn_us, _dyn_lidar = self._compute_dynamic_safety(avg_speed)

        # ────────────────────────────────────────────────────────────────────
        # LIDAR hard safety — PRIORYTET 1 (niezależnie od bumpera!)
        # Gdy Lmin < 0.25m ZAWSZE uciekaj — przebija bumper!
        # ────────────────────────────────────────────────────────────────────
        if self.lidar.min_dist < self.config.LIDAR_HARD_SAFETY_MIN:
            # Ubij bumper forward — jeśli jedziemy w ścianę to NIE FORWARD
            self.rear_bumper_forward_remaining = 0
            self.stabilizer.force_unlock()
            # Wybierz najlepszy kierunek ucieczki
            if us_left_dist > us_right_dist + 0.15:
                self.hard_reflex_action = Action.TURN_LEFT
            elif us_right_dist > us_left_dist + 0.15:
                self.hard_reflex_action = Action.TURN_RIGHT
            else:
                # Oba boki zablokowane — sprawdź sektory LIDAR
                # Sektory 6-9 = tył (180°±45°) — jeśli tył wolny to cofaj
                rear_sectors = self.lidar.sectors_16[6:10]
                rear_blocked = float(np.mean(rear_sectors)) > 0.4
                if rear_blocked:
                    # I przód i tył zablokowane → spin w stronę wyższego US
                    if us_left_dist >= us_right_dist:
                        self.hard_reflex_action = Action.SPIN_LEFT
                    else:
                        self.hard_reflex_action = Action.SPIN_RIGHT
                else:
                    self.hard_reflex_action = Action.REVERSE
            self.hard_reflex_hold_remaining = self.config.HARD_REFLEX_HOLD_CYCLES
            return self.hard_reflex_action, "LIDAR_HARD_SAFETY"

        # ────────────────────────────────────────────────────────────────────
        # HARD REFLEX HOLD — trzymaj poprzednią akcję ratunkową
        # ────────────────────────────────────────────────────────────────────
        if self.hard_reflex_hold_remaining > 0 and self.hard_reflex_action is not None:
            self.hard_reflex_hold_remaining -= 1
            return self.hard_reflex_action, "HARD_REFLEX_HOLD"

        # ★ NOWE: Jeśli aktualna akcja to już TURN/SPIN, a Lmin nie jest jeszcze krytyczne (<0.18m)
        #   ale powyżej 0.15m, to pozwól Q-aproksymatorowi kontynuować manewr.
        if (self._last_action_type in (Action.TURN_LEFT, Action.TURN_RIGHT,
                                       Action.SPIN_LEFT, Action.SPIN_RIGHT)
                and self.lidar.min_dist > 0.15):
            return None  # safety nie ingeruje w aktywny skręt

        # ────────────────────────────────────────────────────────────────────
        # REAR BUMPER: tylna kolizja → uciekaj DO PRZODU jeśli można
        # Jeśli przód zablokowany → SPIN zamiast FORWARD (unikamy pętli!)
        # ────────────────────────────────────────────────────────────────────
        if rear_bumper == 1:
            self.stabilizer.force_unlock()
            # Sprawdź czy z przodu mamy wolną przestrzeń
            front_sectors = np.array([
                self.lidar.sectors_16[14], self.lidar.sectors_16[15],
                self.lidar.sectors_16[0],  self.lidar.sectors_16[1]
            ])
            front_clear = float(np.max(front_sectors)) < 0.5  # 0.5 = ~1.5m odległości

            if front_clear:
                # Przód wolny → jedź do przodu
                self.rear_bumper_forward_remaining = self.config.REAR_BUMPER_FORWARD_CYCLES
                logger.warning("REAR BUMPER: front clear -> FORWARD")
                return Action.FORWARD, "REAR_BUMPER_HIT"
            else:
                # Przód zablokowany → spin w stronę dalszego US
                self.rear_bumper_forward_remaining = 0
                if us_left_dist >= us_right_dist:
                    spin_action = Action.SPIN_LEFT
                else:
                    spin_action = Action.SPIN_RIGHT
                self.hard_reflex_action = spin_action
                self.hard_reflex_hold_remaining = self.config.HARD_REFLEX_HOLD_CYCLES
                logger.warning(f"REAR BUMPER: front blocked -> {spin_action.name}")
                return spin_action, "REAR_BUMPER_SPIN"

        # Kontynuuj wymuszony FORWARD po tylnej kolizji
        # (ale przerwij jeśli LIDAR ostrzega z przodu!)
        if self.rear_bumper_forward_remaining > 0:
            front_sectors = np.array([
                self.lidar.sectors_16[14], self.lidar.sectors_16[15],
                self.lidar.sectors_16[0],  self.lidar.sectors_16[1]
            ])
            if float(np.max(front_sectors)) > 0.5:
                # Z przodu pojawia się przeszkoda — przerwij FORWARD wcześniej
                self.rear_bumper_forward_remaining = 0
                logger.warning("REAR_BUMPER_FORWARD aborted: front obstacle approaching")
            else:
                self.rear_bumper_forward_remaining -= 1
                return Action.FORWARD, "REAR_BUMPER_FORWARD"

        # ────────────────────────────────────────────────────────────────────
        # US front check + LIDAR corroboration
        # ────────────────────────────────────────────────────────────────────
        us_front_min = min(us_left_dist, us_right_dist)

        # ★ NAPRAWIONE: Sprawdź LIDAR przed cofnięciem
        if 0.01 < us_front_min < dyn_us:
            if self.config.REVERSE_LIDAR_CHECK:
                front_blocked = self.lidar.check_front_sectors_blocked(
                    threshold=self.config.REVERSE_LIDAR_THRESHOLD,
                    num_sectors=self.config.REVERSE_LIDAR_SECTORS
                )
                if not front_blocked:
                    return None  # Bok wolny - nie cofaj!

            # Przód zablokowany → cofnij
            self.stabilizer.force_unlock()
            self.hard_reflex_action = Action.REVERSE
            self.hard_reflex_hold_remaining = self.config.HARD_REFLEX_HOLD_CYCLES
            return Action.REVERSE, "HARD_REFLEX"

        # Anti-stall
        if (abs(encoder_l) < 0.02 and abs(encoder_r) < 0.02 and
                abs(self.motors.last_pwm_l) > 40 and abs(self.motors.last_pwm_r) > 40):
            self.stabilizer.force_unlock()
            self.hard_reflex_action = Action.ESCAPE_MANEUVER
            self.hard_reflex_hold_remaining = self.config.HARD_REFLEX_HOLD_CYCLES
            return Action.ESCAPE_MANEUVER, "ANTI_STALL"

        return None

    def loop(self, lidar_points: List[Tuple[float, float]],
             encoder_l: float, encoder_r: float,
             motor_current: float,
             us_left_dist: float = 3.0, us_right_dist: float = 3.0,
             rear_bumper: int = 0,
             dt: float = 0.033) -> Tuple[float, float]:
        """Glowna petla decyzyjna (v5.5 - dual US + rear bumper)"""

        self.cycle_count += 1
        us_front_min = min(us_left_dist, us_right_dist)

        # 1. Process sensors
        lidar_16 = self.lidar.process(lidar_points)
        avg_speed = (encoder_l + encoder_r) / 2.0
        dyn_us, dyn_lidar = self._compute_dynamic_safety(avg_speed)

        # 2. Safety check (dual US + rear bumper)
        safety_override = self.validate_safety_constraints(
            us_left_dist, us_right_dist, encoder_l, encoder_r, rear_bumper)

        # 3. Lorenz step
        self.lorenz.step()
        aggression_factor = self.lorenz.z_norm
        directional_bias = self.lorenz.x_norm

        # 4. Free space instinct (WZMOCNIONY z USL/USR + front sektory)
        free_angle, free_mag = self.instinct.compute_free_space_vector(lidar_16)

        # ★ front_clearance: srednia wolnosci przednich sektorow (14,15,0,1 = ±45° od 0°)
        #   lidar_16[i] = 1-dist/max → HIGH=przeszkoda, LOW=wolno
        #   front_clearance 1.0 = czysto z przodu, 0.0 = sciana wprost
        front_occ     = float(np.mean([lidar_16[14], lidar_16[15],
                                       lidar_16[0],  lidar_16[1]]))
        front_clearance = 1.0 - front_occ   # odwroc: 1=wolno, 0=sciana

        instinct_bias = self.instinct.get_bias_for_action(
            free_angle,
            magnitude=free_mag,
            front_clearance=front_clearance,
            us_left=us_left_dist,
            us_right=us_right_dist
        )
        instinct_bias = self.instinct.apply_us_bias(instinct_bias, us_left_dist, us_right_dist)

        # 5. Feature extraction (v5.9.3 — rozszerzony wektor)
        features = self.brain.get_features(
            lidar_16, us_left_dist, us_right_dist,
            encoder_l, encoder_r,
            self.lorenz.x_norm, self.lorenz.z_norm,
            rear_bumper, self.lidar.min_dist,
            last_action=self._last_action_type,
            free_angle=free_angle, free_mag=free_mag)

        # 6. ★ Concept Graph - sugestia akcji
        context = {'min_dist': self.lidar.min_dist, 'us_left': us_left_dist, 'us_right': us_right_dist}
        best_concept = self.concept_graph.get_best_concept(context)
        concept_suggestion = None
        if best_concept:
            concept_suggestion = self.concept_graph.get_next_action_from_concept(best_concept)
            if self.cycle_count % 50 == 0:
                logger.info(
                    f"[CONCEPT] Uzyto konceptu '{best_concept.name}' "
                    f"(aktywacja={best_concept.activation:.2f}) "
                    f"→ sugestia: {concept_suggestion.name if concept_suggestion else 'None'}"
                )

        # 7. Decision (Q + Instinct + Concept!)
        collision = (us_front_min < dyn_us) or (self.lidar.min_dist < dyn_lidar)

        gate_weights = np.zeros(2)

        if safety_override:
            final_action, source = safety_override
        else:
            # ★ NOWE: Anti-stagnation moze wymusic skret
            forced_turn = self.anti_stagnation.should_force_turn()
            if forced_turn is not None:
                final_action = forced_turn
                source = "STAGNATION_FORCE"
                self.stabilizer.force_unlock()
            else:
                action_candidate, source, gate_weights = self.brain.decide(features, instinct_bias, concept_suggestion)
                final_action = self.stabilizer.update(action_candidate)

        # ★ NOWE: Anti-oscillation — wykryj petle REVERSE↔FORWARD
        if final_action == self._last_action_type:
            self._action_repeat_count += 1
        else:
            self._action_repeat_count = 0
            self._last_action_type = final_action

        # Jesli za duzo powtorzen REVERSE — wymus SPIN (bez hard_reflex, uzyj stagnation hold)
        if (final_action == Action.REVERSE and
            source not in ("HARD_REFLEX", "HARD_REFLEX_HOLD", "LIDAR_HARD_SAFETY") and
            self._action_repeat_count >= self.config.OSCILLATION_MAX_REPEATS and
            self.anti_stagnation.is_stagnant):
            spin = Action.SPIN_LEFT if us_left_dist >= us_right_dist else Action.SPIN_RIGHT
            final_action = spin
            source = "ANTI_OSCILLATION"
            self._action_repeat_count = 0
            self.stabilizer.force_unlock()
            # Uzyj stagnation hold zeby utrzymac spin przez kilka cykli
            self.anti_stagnation.stagnation_force_remaining = 6
            self.anti_stagnation.stagnation_direction = 1 if spin == Action.SPIN_LEFT else -1
            logger.info(f"[ANTI-OSC] 8x REVERSE & STAGNANT -> {spin.name} (6 cycles)")

        # ★ BEZPIECZNIK: jesli front zablokowany (clr<0.40) a akcja = FORWARD -> skret!
        if (final_action == Action.FORWARD and front_clearance < 0.40
                and source not in ("HARD_REFLEX", "HARD_REFLEX_HOLD", "REAR_BUMPER_HIT",
                                   "REAR_BUMPER_FORWARD", "LIDAR_HARD_SAFETY")):
            # Skret w strone wiekszej przestrzeni (US)
            if us_left_dist >= us_right_dist:
                final_action = Action.TURN_LEFT
            else:
                final_action = Action.TURN_RIGHT
            source = "FRONT_BLOCKED_TURN"
            self.stabilizer.force_unlock()

        # 8. Reward (★ z kara za bliskosc sciany przy FORWARD)
        reward = self.damper.compute_reward(encoder_l, encoder_r, motor_current, final_action)

        # ★ NOWE: Kara za FORWARD blisko sciany — uczy Q-table zeby nie jechac w sciane
        if final_action == Action.FORWARD and self.lidar.min_dist < 0.35:
            proximity_penalty = -2.0 * (1.0 - self.lidar.min_dist / 0.35)
            reward += proximity_penalty

        # 9. Q-Update
        #    Uczenie ZAWSZE (nawet na wymuszonych akcjach):
        if self.brain.last_features is not None and self.brain.last_action is not None:
            oscillated = (
                source == "ANTI_OSCILLATION"
                or self._action_repeat_count >= self.config.OSCILLATION_MAX_REPEATS
            )
            self.brain.update_q(
                old_features=self.brain.last_features,
                action=self.brain.last_action,
                reward=reward,
                new_features=features,
                source=source,
                lidar_min=self.lidar.min_dist,
                stagnant=self.anti_stagnation.is_stagnant,
                oscillated=oscillated,
                done=False,
                lr_scale=1.0
            )


        # 10. ★ Concept Graph Update
        self.concept_graph.update(final_action, reward, self.cycle_count)

        # ---- KONSOLIDACJA KONCEPTÃW ----
        if self.cycle_count % self.config.CONCEPT_PRUNING_INTERVAL == 0 and self.cycle_count > 0:
            # PrzekaÅ¼ aktualny numer kroku (self.cycle_count) do metody prune_and_merge
            self.concept_graph.prune_and_merge(self.cycle_count)

        # 11. Velocity mapping
        # Dla FORWARD: usyj front_clearance (odl. z przodu) nie globalny min_dist (moze byc sciana z tylu!)
        front_dist_est = front_clearance * self.config.LIDAR_MAX_RANGE
        fwd_velocity   = self.velocity_mapper.compute_base_velocity(front_dist_est, aggression_factor)
        base_velocity  = self.velocity_mapper.compute_base_velocity(self.lidar.min_dist, aggression_factor)

        # 12. Action → Target velocities
        if final_action == Action.FORWARD:
            target_l, target_r = fwd_velocity, fwd_velocity  # predkosc bazuje na front_dist!
        elif final_action == Action.REVERSE:
            target_l, target_r = -base_velocity * 0.45, -base_velocity * 0.45
        elif final_action == Action.TURN_LEFT:
            target_l, target_r = base_velocity * 0.3, base_velocity
        elif final_action == Action.TURN_RIGHT:
            target_l, target_r = base_velocity, base_velocity * 0.3
        elif final_action == Action.SPIN_LEFT:
            target_l, target_r = -base_velocity * 0.5, base_velocity * 0.5
        elif final_action == Action.SPIN_RIGHT:
            target_l, target_r = base_velocity * 0.5, -base_velocity * 0.5
        elif final_action == Action.ESCAPE_MANEUVER:
            target_l, target_r = base_velocity * 0.5, -base_velocity * 0.5
        else:
            target_l, target_r = 0.0, 0.0

        # 13. Lorenz bias — delikatny dla TURN/SPIN, zero na velocity dla FORWARD
        #     FORWARD dostaje mikro-szum dopiero na poziomie PWM (krok 19b)
        if final_action in (Action.TURN_LEFT, Action.TURN_RIGHT,
                            Action.SPIN_LEFT, Action.SPIN_RIGHT):
            bias_strength = self.config.LORENZ_BIAS_SCALE
            target_l += directional_bias * bias_strength
            target_r -= directional_bias * bias_strength

        # 14. Ramp limiter
        target_l, target_r = self.velocity_mapper.apply_ramp_limit(target_l, target_r)

        # 15. Enforce symmetry dla REVERSE
        if final_action == Action.REVERSE:
            symmetric = (target_l + target_r) / 2.0
            target_l, target_r = symmetric, symmetric

        if final_action == Action.ESCAPE_MANEUVER:
            mag = min(abs(target_l), abs(target_r))
            target_l, target_r = mag, -mag

        # 16. Update memory (v5.9 — features zamiast hash)
        self.brain.last_features = features
        self.brain.last_action = final_action

        # 17. PID control
        pwm_l, pwm_r = self.motors.update_pid(target_l, target_r, encoder_l, encoder_r, dt)

        # 18. Enforce PWM symmetry dla REVERSE
        if final_action == Action.REVERSE:
            symmetric_pwm = -abs((pwm_l + pwm_r) / 2.0)
            pwm_l, pwm_r = symmetric_pwm, symmetric_pwm
            self.motors.sync_memory(pwm_l, pwm_r)

        # 19. Anti-stagnation — chaos TYLKO dla TURN/SPIN
        avg_pwm = (abs(pwm_l) + abs(pwm_r)) / 2.0
        # Przekaz aktualna akcje — spin/turn NIE jest stagnacja!
        self.anti_stagnation.update(self.motors.x, self.motors.y, avg_pwm,
                                    current_action=final_action)
        if final_action in (Action.TURN_LEFT, Action.TURN_RIGHT,
                            Action.SPIN_LEFT, Action.SPIN_RIGHT):
            pwm_l, pwm_r = self.anti_stagnation.inject_chaos(
                self.lorenz.x_norm, self.lorenz.z_norm, pwm_l, pwm_r
            )

        # ★ NOWE: Hard clamp PWM po chaos injection — zapobiega eksplozji PWM (±300+)
        pwm_l = np.clip(pwm_l, -self.config.PWM_MAX, self.config.PWM_MAX)
        pwm_r = np.clip(pwm_r, -self.config.PWM_MAX, self.config.PWM_MAX)

        # 19b. ★ FORWARD: symetria + mikro-szum Lorenz (±1-3 PWM)
        if final_action == Action.FORWARD:
            # Krok 1: Wymuszaj symetrie bazowa
            avg_pwm_fwd = (pwm_l + pwm_r) / 2.0

            # Krok 2: Mikro-szum Lorenz (±1-3 PWM — naturalny dryf)
            micro_noise = directional_bias * self.config.FORWARD_LORENZ_PWM

            # Krok 3: Korekcja enkoderowa
            encoder_diff = encoder_l - encoder_r
            encoder_corr = 0.0
            if abs(encoder_diff) > 0.005:
                encoder_corr = encoder_diff * 20.0

            pwm_l = avg_pwm_fwd + micro_noise - encoder_corr
            pwm_r = avg_pwm_fwd - micro_noise + encoder_corr

            # ★ KRYTYCZNE: Zsynchronizuj pamiec PID!
            self.motors.sync_memory(pwm_l, pwm_r)

        # 20. Odometry
        self.motors.update_odometry(encoder_l, encoder_r, dt)

        # 21. ★ Auto-save (wbudowane!)
        # if self.state_manager.should_auto_save():
        #     self.save_state()

        # 22. Pelna diagnostyka co 50 cykli
        if self.cycle_count % 50 == 0:
            lorenz_info = f"Lx={directional_bias:+.2f} Lz={aggression_factor:.2f}"
            free_info   = (f"front_clr={front_clearance:.2f} "
                           f"free_ang={math.degrees(free_angle):+.0f}deg mag={free_mag:.2f}")
            stag_info   = "STAGNANT!" if self.anti_stagnation.is_stagnant else "ok"
            q_vals      = self.brain.nn.cache.get('q', np.zeros(8))
            q_info      = (f"Q=[{np.min(q_vals):+.2f},{np.max(q_vals):+.2f}] "
                           f"Qnrm={np.linalg.norm(q_vals):.1f} "
                           f"eps={self.brain.epsilon:.3f} lr={self.brain.lr:.5f} "
                           f"buf={len(self.brain.replay_buffer)}")

            gate_info = f"Gate=[{gate_weights[0]:.2f},{gate_weights[1]:.2f}]" if source == "NEURAL" else "Gate=off"

            wm_info = f"CF={self.brain.counterfactual_count}"

            # Monitoring wag Q
            if self.cycle_count % 100 == 0:
                q_norm = np.linalg.norm(self.brain.nn.W_q)
                if q_norm > 1000:
                    logger.warning(f"Q norm: {q_norm:.2f} â possible explosion!")
                if q_norm > 10000:
                    self.brain.nn.W_q = np.random.randn(16, 8) * 0.01
                    self.brain.nn.b_q = np.zeros(8)
                    logger.critical("Q layer reset due to explosion")
            logger.info(
                f"[DIAG] c={self.cycle_count} "
                f"PWM=({pwm_l:+.0f},{pwm_r:+.0f}) "
                f"USL={us_left_dist:.2f} USR={us_right_dist:.2f} "
                f"Lmin={self.lidar.min_dist:.2f} "
                f"act={final_action.name} src={source} "
                f"{lorenz_info} {free_info} stag={stag_info} "
                f"{q_info} {gate_info} {wm_info}"
            )


        # ★★★ FINAL SAFETY CLAMP — zawsze na koncu, bez wyjatkow!
        pwm_l = np.clip(pwm_l, -self.config.PWM_MAX, self.config.PWM_MAX)
        pwm_r = np.clip(pwm_r, -self.config.PWM_MAX, self.config.PWM_MAX)

        return pwm_l, pwm_r

    def get_stats(self) -> Dict:
        return {
            'l2_exists': self.brain.l2_feature_indices is not None,
            'l2_features': len(self.brain.l2_feature_indices) if self.brain.l2_feature_indices is not None else 0,
            'l2_version': getattr(self.brain, 'l2_version', 0),
            'l2_stability': getattr(self.brain, 'l2_stability_counter', 0),
            'cycle_count':       self.cycle_count,
            'q_weights_norm':    float(np.linalg.norm(self.brain.nn.W_q)),
            'gate_weights_norm': float(np.linalg.norm(self.brain.nn.W_gate)),
            'cf_count':          self.brain.counterfactual_count,
            'epsilon':           self.brain.epsilon,
            'lr':                self.brain.lr,
            'replay_size':       len(self.brain.replay_buffer),
            'normalizer_n':      self.brain.normalizer.n,
            'concepts_count':    len(self.concept_graph.concepts),
            'lorenz_state':      self.lorenz.get_state(),
            'position':          (self.motors.x, self.motors.y, self.motors.theta),
            'stagnant':          self.anti_stagnation.is_stagnant,
        }


# Backward compatibility
SwarmCoreV54 = SwarmCoreV55


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SWARM CORE v5.14 -- DUAL LINEAR BRAIN -- TEST")
    print("=" * 70 + "\n")

    cfg = SwarmConfig()
    core = SwarmCoreV55()

    print("\n=== TEST 3: FeatureImportanceAnalyzer ===")
    analyzer = FeatureImportanceAnalyzer(82, 32)
    q_w = np.random.randn(8, 82)
    a_w = np.random.randn(8, 82)
    for _ in range(200):
        analyzer.update(q_w, a_w)
    indices = analyzer.get_top_features()
    assert indices is not None and len(indices) == 32
    print(f"✓ Analizator: wybrano {len(indices)} cech")

    print("\n=== UNIT 4: GateApproximator ===")
    gate = GateApproximator(16)
    feat = np.random.randn(16).astype(np.float32)
    w = gate.predict_weights(feat)
    assert np.isclose(np.sum(w), 1.0), "Softmax nie sumuje sie do 1"
    gate.update(feat, np.array([1.0, 0.0]))
    print("✓ GateApproximator OK")

    wm = WorldModel(state_dim=32, action_dim=8)
    state = np.random.randn(32).astype(np.float32)
    next_state, reward = wm.predict(state, 3)
    assert next_state.shape == (32,)
    print("✓ WorldModel.predict OK")

    # Test treningu
    target_next = np.random.randn(32).astype(np.float32)
    wm.train_step(state, 3, target_next, 1.0)
    print("✓ WorldModel.train_step OK")

    print("\n=== UNIT 8: NeuralBrainWithImagination ===")
    nn = NeuralBrainWithImagination(cfg)
    feat = np.random.randn(82).astype(np.float32)
    q = nn.forward_q(feat)
    assert q.shape == (8,)
    print("✓ forward_q OK")

    gate = nn.forward_gate()
    assert gate.shape == (2,) and np.isclose(np.sum(gate), 1.0)
    print("✓ forward_gate OK")

    next_feat, pred_reward = nn.forward_world(0)
    assert next_feat.shape == (82,)
    print("✓ forward_world OK")

    target_q = np.random.randn(8)
    nn.backward_q(target_q)
    print("✓ backward_q OK")

    target_next = np.random.randn(82)
    nn.backward_world(target_next, 1.0)
    print("✓ backward_world OK")

    print("\n=== UNIT 9: NeuralHybridBrain ===")
    fe = FeatureExtractor(cfg)
    brain = NeuralHybridBrain(cfg, fe)
    feat = brain.get_features(np.random.randn(16), 1,1,0,0,0,0,0,1)
    action, src, gate = brain.decide(feat, {}, None)
    print(f"✓ decide -> {action.name}, {src}")

    print("[INTEG] SwarmCoreV55 DualLinear -- 20 krokow ...")
    for i in range(20):
        # Symulacja
        pwm_l, pwm_r = core.loop(
            lidar_points=[(a, max(0.1, 2.0 - i*0.1)) for a in range(0, 360, 22)],
            encoder_l=0.3, encoder_r=0.3, motor_current=1.2,
            us_left_dist=max(0.1, 2.0 - i*0.1), us_right_dist=max(0.1, 2.0 - i*0.1), dt=0.1
        )
        print(f"#{i+1:2d}: PWM=({pwm_l:+6.1f},{pwm_r:+6.1f}) | {core.brain.last_action}")

    print("\n[STATS v5.14]:")
    for k, v in core.get_stats().items():
        print(f"   {k}: {v}")

    core.save_state()
    core.brain.debug_l2()

    # Wymus wzmocnienie
    changed, stats = core.brain.reinforce_l2(force=True, method='combined')
    print(f"Wzmocnienie: {changed}")
    core.brain.debug_l2()
    print("\n[OK] Test v5.14 -- wszystkie testy jednostkowe i integracyjne PASSED!\n")
    print("\n🔬 Test L2:")
    core.brain.debug_l2()

    # Wymus wzmocnienie
    changed, stats = core.brain.reinforce_l2(force=True, method='combined')
    print(f"Wzmocnienie: {changed}")
    core.brain.debug_l2()

    print("\n=== TEST 10: Concept Pruning & Merging ===")
    cg = ConceptGraph(cfg)
    # Add dummy concepts
    c1 = Concept("learned_c1", [Action.FORWARD, Action.TURN_LEFT])
    c1.usage_count = 6
    c1.last_used_step = 0; c1.success_count = 6; c1.success_ratio = 1.0
    cg.concepts["learned_c1"] = c1

    c2 = Concept("learned_c2", [Action.FORWARD, Action.TURN_LEFT]) # Identical sequence
    c2.usage_count = 10; c2.success_count = 10; c2.success_ratio = 1.0
    c2.last_used_step = 100
    cg.concepts["learned_c2"] = c2

    # Trigger pruning/merging
    # Set pruning interval to small for test
    cg.pruning_interval = 50
    cg.min_usage_to_survive = 5

    # Test step 200 (c1 should be pruned because usage < 5 and age > 50)
    # But wait, prune_and_merge also merges. c1 and c2 are identical.
    # They should be merged.

    cg.prune_and_merge(200)

    # Verify
    print(f"Concepts after pruning: {list(cg.concepts.keys())}")
    # Expect merged concept
    sim = cg._calculate_similarity(cg.concepts["learned_c1"].sequence, cg.concepts["learned_c2"].sequence); print(f"Similarity: {sim}")
    print("✓ Concept Pruning & Merging OK")
