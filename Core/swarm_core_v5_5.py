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
    ROBOT_WIDTH: float = 0.32        # szerokość obudowy [m]
    ROBOT_LENGTH: float = 0.25       # długość obudowy [m]
    ROBOT_SHAPE: str = "trapez"  # ksztalt
    ENCODER_TICKS_PER_M: float = 5000.0 # Przykladowa wartosc, do kalibracji
    ROBOT_WHEEL_SPAN: float = 0.32   # rozstaw kół [m] (zewn. krawędzie)
    WHEEL_BASE: float = 0.32         # rozstaw kół do odometrii [m]
    ROBOT_HALF_WIDTH: float = 0.16   # połowa rozstawu kół
    US_FORWARD_OFFSET: float = 0.29  # odległość US od osi obrotu [m]
    MAX_SPEED_MPS: float = 0.5

    # Dystanse (NAPRAWIONE - 10cm target)
    # SAFETY - minimalne progi (naprawdę blisko!)
    US_SAFETY_DIST: float = 0.10        # 10 cm - dopiero tutaj reaguj!
    US_TARGET_DIST: float = 0.10        # 10 cm - cel jazdy

    LIDAR_SAFETY_RADIUS: float = 0.10   # 10 cm - dopiero tutaj reaguj!
    LIDAR_MAX_RANGE: float = 3.0

    # Dynamiczne safety (zależne od prędkości)
    SAFETY_DIST_SPEED_SCALE: float = 0.1   # Mniejsze skalowanie
    SAFETY_US_MIN: float = 0.08            # 8 cm - absolutne minimum
    SAFETY_US_MAX: float = 0.15            # 15 cm - maximum przy dużej prędkości
    SAFETY_LIDAR_MIN: float = 0.08         # 8 cm
    SAFETY_LIDAR_MAX: float = 0.15         # 15 cm

    # HARD SAFETY - dopiero przy 10 cm!
    LIDAR_HARD_SAFETY_MIN: float = 0.10    # 10 cm - dopiero tutaj panika!
    HARD_REFLEX_HOLD_CYCLES: int = 2       # Krótko trzymaj akcję awaryjną

    # Rear bumper
    REAR_BUMPER_FORWARD_CYCLES: int = 3  # Ile cykli FORWARD po kolizji tylnej

    # Korekcja jazdy prostej (encoder-based)
    FORWARD_ENCODER_CORRECTION: float = 0.5  # Siła korekcji enkoderowej
    FORWARD_LORENZ_PWM: float = 2.5       # Max ±PWM szum Lorenz przy FORWARD
    FORWARD_CHAOS_DAMPEN: float = 0.0      # Chaos inject = 0 dla FORWARD

    # Cofanie z LIDAR (NAPRAWIONE)
    REVERSE_LIDAR_CHECK: bool = True
    REVERSE_LIDAR_SECTORS: int = 4
    REVERSE_LIDAR_THRESHOLD: float = 0.10  # 10 cm - sprawdzaj LIDAR

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
                free_angle: float = 0.0, free_mag: float = 0.0,
                slip_ratio: float = 0.0, is_stalled: bool = False,
                traj_correction: float = 0.0, stall_intensity: float = 0.0) -> np.ndarray:
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

        # Krok 2: Używamy DualLinearApproximator zamiast NeuralBrain
        self.q_approx = DualLinearApproximator(
            n_features=self.n_features,
            n_actions=self.n_actions,
            learning_rate=config.LEARNING_RATE,
            avoidance_penalty=config.AVOIDANCE_PENALTY,
            weight_clip=10.0
        )

        self.lr = config.NN_LEARNING_RATE

        # Standardowe komponenty
        self.normalizer = RunningNormalizer(self.n_features)
        self.replay_buffer = ReplayBuffer(capacity=config.REPLAY_BUFFER_CAPACITY)
        self.epsilon = config.EPSILON

        self.last_features = None
        self.last_action = None

        logger.info(f"NeuralHybridBrain: DualLinearApproximator initialized")

    def get_features(self, lidar_16, us_left, us_right, encoder_l, encoder_r,
                     lorenz_x, lorenz_z, rear_bumper, min_dist, last_action=None,
                     free_angle=0.0, free_mag=0.0,
                     slip_ratio=0.0, is_stalled=False, traj_correction=0.0, stall_intensity=0.0) -> np.ndarray:
        raw = self.feature_extractor.extract(
            lidar_16, us_left, us_right, encoder_l, encoder_r,
            lorenz_x, lorenz_z, rear_bumper, min_dist, last_action, free_angle, free_mag,
            slip_ratio, is_stalled, traj_correction, stall_intensity
        )
        self.normalizer.update(raw)
        return self.normalizer.normalize(raw)

    def decide(self, features: np.ndarray, instinct_bias: Dict[Action, float],
               concept_suggestion: Optional[Action]) -> Tuple[Action, str, np.ndarray]:

        if random.random() < self.epsilon:
            return random.choice(self.actions_list), "EXPLORE", np.zeros(2)

        combined_q = self.q_approx.predict(features)   # już z uwzględnieniem A
        scores = combined_q.copy()

        for i, action in enumerate(self.actions_list):
            scores[i] += instinct_bias.get(action, 0.0) * 4.0
            if action == concept_suggestion:
                scores[i] += 1.5

        best_idx = int(np.argmax(scores))
        return self.actions_list[best_idx], "Q_APPROX+INST+CONCEPT", np.zeros(2)

    def is_bad_state(self, reward: float, source: str, action: Action,
                     lidar_min: float, stagnant: bool, oscillated: bool) -> Tuple[bool, float]:
        # Kolizja / twardy odruch (ale nie gdy sam REVERSE)
        if source in ("LIDAR_HARD_SAFETY", "HARD_REFLEX") and action != Action.REVERSE:
            return True, 2.0   # bardzo źle
        # Stagnacja (ale nie podczas skretu)
        if stagnant and action in (Action.FORWARD, Action.REVERSE):
            return True, 1.5
        # Oscylacja
        if oscillated:
            return True, 1.2
        # Niska nagroda
        if reward < -1.5:
            return True, 1.0
        # Jazda do przodu zbyt blisko ściany (ale jeszcze nie kolizja)
        if action == Action.FORWARD and lidar_min < 0.2:
            return True, 0.8

        # Domyslnie – dobre doswiadczenie
        return False, 0.0

    def update_q(self, old_features: np.ndarray, action: Action,
                 reward: float, new_features: np.ndarray,
                 source: str, lidar_min: float,
                 stagnant: bool, oscillated: bool,
                 done: bool = False, lr_scale=1.0):

        action_idx = self.actions_list.index(action)

        # Czy to złe doświadczenie?
        is_bad, bad_target = self.is_bad_state(reward, source, action,
                                                lidar_min, stagnant, oscillated)

        if is_bad:
            # Aktualizuj macierz unikania
            self.q_approx.update_a(old_features, action_idx, bad_target)
        else:
            # Normalna aktualizacja Q (TD)
            if done:
                target_q = reward
            else:
                next_q = self.q_approx.predict_q(new_features)
                max_next_q = float(np.max(next_q))
                target_q = reward + self.config.DISCOUNT_FACTOR * max_next_q
            self.q_approx.update_q(old_features, action_idx, target_q)

        # Decay epsilon i LR
        self.epsilon = max(self.config.EPSILON_MIN,
                           self.epsilon * self.config.EPSILON_DECAY)
        self.lr = max(self.config.LR_MIN,
                      self.lr * self.config.LR_DECAY)
        self.q_approx.set_learning_rate(self.lr)

class StuckDetector:
    """
    Wykrywa czy robot jest zablokowany.
    Koła się kręcą (enkodery > 0), ale odległość z przodu nie maleje.
    """
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.front_dist_history = deque(maxlen=10)
        self.stuck_count = 0

    def update(self, front_dist: float, enc_l: float, enc_r: float) -> bool:
        self.front_dist_history.append(front_dist)

        # Czy koła się kręcą?
        wheels_moving = abs(enc_l) > 0.01 or abs(enc_r) > 0.01

        if not wheels_moving:
            self.stuck_count = 0
            return False

        # Czy odległość z przodu się zmienia?
        if len(self.front_dist_history) == 10:
            dist_change = max(self.front_dist_history) - min(self.front_dist_history)
            if dist_change < 0.02:  # Zmiana < 2cm przez 10 cykli
                self.stuck_count += 1
                if self.stuck_count > 5:  # 5 cykli potwierdzenia
                    return True
            else:
                self.stuck_count = max(0, self.stuck_count - 1)

        return False

class MovementTracker:
    """
    Śledzi sekwencje ruchu i nagradza za rzeczywiste przemieszczenie.

    ZASADA DZIAŁANIA:
    1. Zapamiętujemy sekwencje akcji (np. FORWARD, FORWARD, REVERSE)
    2. Mierzymy czy robot faktycznie zmienił pozycję w tym czasie
    3. Jeśli sekwencja nie dała ruchu → kara
    4. Jeśli dała ruch → nagroda (proporcjonalna do przebytej drogi)

    WAŻNE: Nie karzemy pojedynczych akcji, tylko BEZSKUTECZNE SEKWENCJE!
    """

    def __init__(self, config: SwarmConfig):
        self.config = config
        self.action_buffer = deque(maxlen=10)  # ostatnie 10 akcji
        self.position_buffer = deque(maxlen=10)  # pozycje w tych samych momentach

        # Liczniki do detekcji pętli
        self.forward_count = 0
        self.reverse_count = 0
        self.sequence_start_pos = None
        self.sequence_start_step = 0
        self.current_sequence = []

        # Statystyki
        self.total_distance = 0.0
        self.effective_moves = 0
        self.wasted_moves = 0

    def update(self, action: Action, x: float, y: float, step: int):
        """
        Aktualizuj tracker po każdej akcji.
        """
        self.action_buffer.append(action)
        self.position_buffer.append((x, y))
        self.current_sequence.append(action)

        # Zliczaj FORWARD i REVERSE w sekwencji
        if action == Action.FORWARD:
            self.forward_count += 1
        elif action == Action.REVERSE:
            self.reverse_count += 1

        # Inicjalizuj początek sekwencji
        if self.sequence_start_pos is None:
            self.sequence_start_pos = (x, y)
            self.sequence_start_step = step

        # Co 5 akcji lub gdy zmienia się typ (FORWARD->REVERSE)
        if len(self.current_sequence) >= 5 or self._is_pattern_break():
            self._evaluate_sequence(x, y, step)

    def _is_pattern_break(self) -> bool:
        """
        Czy nastąpiło przerwanie wzorca?
        Wykrywa zmianę FORWARD<->REVERSE.
        """
        if len(self.current_sequence) < 2:
            return False

        last = self.current_sequence[-1]
        prev = self.current_sequence[-2]

        # FORWARD -> REVERSE lub REVERSE -> FORWARD to potencjalna pętla
        if (last == Action.FORWARD and prev == Action.REVERSE) or            (last == Action.REVERSE and prev == Action.FORWARD):
            return True

        return False

    def _evaluate_sequence(self, current_x: float, current_y: float, step: int):
        """
        Oceń całą sekwencję od ostatniego resetu.
        """
        if self.sequence_start_pos is None:
            return

        # Oblicz przemieszczenie w tej sekwencji
        start_x, start_y = self.sequence_start_pos
        distance = math.sqrt((current_x - start_x)**2 + (current_y - start_y)**2)

        # Długość sekwencji w krokach
        sequence_length = step - self.sequence_start_step
        if sequence_length == 0: sequence_length = 1

        # Czy sekwencja była skuteczna?
        if distance > 0.05:  # Przesunął się o >5cm
            # NAGRODA! Im dalej zaszedł, tym większa
            reward_per_step = distance / sequence_length
            self.effective_moves += 1

            logger.debug(f"SEQUENCE SUCCESS: {distance:.2f}m w {sequence_length} krokach")

            # Bonus za długą, skuteczną sekwencję
            if sequence_length >= 5 and distance > 0.2:
                logger.info(f"🏆 Długa skuteczna sekwencja! +{distance*10:.1f} bonus")

        else:
            # KARA! Wykonał akcje ale nie ruszył się
            self.wasted_moves += 1

            # Sprawdź czy to pętla FORWARD-REVERSE
            if self.forward_count >= 2 and self.reverse_count >= 2:
                logger.warning(f"⚠️ PĘTLA FORWARD-REVERSE: {self.forward_count}xF, {self.reverse_count}xR bez ruchu")

        # Resetuj liczniki sekwencji
        self.sequence_start_pos = (current_x, current_y)
        self.sequence_start_step = step
        self.current_sequence = []
        self.forward_count = 0
        self.reverse_count = 0

    def get_sequence_reward(self) -> float:
        """
        Zwraca nagrodę za bieżącą sekwencję (do Q-learning).
        """
        if self.sequence_start_pos is None:
            return 0.0

        # Jeśli sekwencja jest długa i nieskuteczna - kara
        if len(self.current_sequence) >= 8:
            # Sprawdź czy to pętla
            forward_ratio = self.forward_count / max(1, len(self.current_sequence))
            reverse_ratio = self.reverse_count / max(1, len(self.current_sequence))

            # Jeśli dużo FORWARD i REVERSE bez innych akcji
            if forward_ratio > 0.3 and reverse_ratio > 0.3:
                return -2.0  # DUŻA KARA za pętlę

        return 0.0

    def is_oscillating(self) -> bool:
        """
        Czy robot właśnie oscyluje (FORWARD/REVERSE bez ruchu)?
        """
        if len(self.current_sequence) < 4:
            return False

        # Ostatnie 4 akcje
        last4 = list(self.current_sequence)[-4:]

        # Sprawdź wzór F,R,F,R lub R,F,R,F
        pattern1 = [Action.FORWARD, Action.REVERSE, Action.FORWARD, Action.REVERSE]
        pattern2 = [Action.REVERSE, Action.FORWARD, Action.REVERSE, Action.FORWARD]

        if last4 == pattern1 or last4 == pattern2:
            # Sprawdź czy faktycznie stoi w miejscu
            if len(self.position_buffer) >= 4:
                positions = list(self.position_buffer)[-4:]
                x_coords = [p[0] for p in positions]
                y_coords = [p[1] for p in positions]

                # Jeśli pozycja się nie zmienia
                if max(x_coords) - min(x_coords) < 0.03 and                    max(y_coords) - min(y_coords) < 0.03:
                    return True

        return False

    def get_stats(self) -> dict:
        """Statystyki dla diagnostyki"""
        return {
            'effective_moves': self.effective_moves,
            'wasted_moves': self.wasted_moves,
            'efficiency': self.effective_moves / max(1, self.effective_moves + self.wasted_moves),
            'current_sequence_len': len(self.current_sequence)
        }
class SensorFusion:
    """
    Fuzja sensorów: US (15 stopni) + LIDAR (360 stopni).
    Zwraca jedno źródło prawdy o otoczeniu.
    """
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.front_dist = 0.0
        self.rear_dist = 0.0
        self.left_dist = 0.0
        self.right_dist = 0.0

    def update(self, us_left: float, us_right: float, lidar_16: np.ndarray):
        # US: 15 stopni od osi pojazdu
        # LIDAR: sektory 14,15 (lewy przód), 0,1 (prawy przód)

        # Front dist = min(US, LIDAR_front)
        us_min = min(us_left, us_right)

        # LIDAR front (sektory 15, 0) - najbardziej centralne
        lidar_front_central = min(lidar_16[15], lidar_16[0])
        # Skalowanie lidar_16 (jest 1-dist/max) na metry
        # 1.0 = 0m, 0.0 = max_range
        # dist = (1.0 - val) * max_range
        lidar_dist_m = (1.0 - lidar_front_central) * self.config.LIDAR_MAX_RANGE

        self.front_dist = min(us_min, lidar_dist_m)

        # Rear (sektory 7, 8)
        lidar_rear = min(lidar_16[7], lidar_16[8])
        self.rear_dist = (1.0 - lidar_rear) * self.config.LIDAR_MAX_RANGE

        # Left (sektory 11, 12)
        lidar_left = min(lidar_16[11], lidar_16[12])
        self.left_dist = (1.0 - lidar_left) * self.config.LIDAR_MAX_RANGE

        # Right (sektory 3, 4)
        lidar_right = min(lidar_16[3], lidar_16[4])
        self.right_dist = (1.0 - lidar_right) * self.config.LIDAR_MAX_RANGE

class BumperSystem:
    """
    Obsługa fizycznych zderzaków (damperów).
    Absolutny priorytet.
    """
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.escape_sequence = 0
        self.escape_action = None

    def check_collision(self, rear_bumper: int) -> Optional[Action]:
        # Jeśli trwa ucieczka
        if self.escape_sequence > 0:
            self.escape_sequence -= 1
            return self.escape_action

        # Wykrycie kolizji
        if rear_bumper == 1:
            # Kolizja z tyłu -> uciekaj do przodu
            logger.warning("BUMPER: Kolizja z tyłu!")
            self.escape_sequence = 5 # 5 cykli
            self.escape_action = Action.FORWARD
            return Action.FORWARD

        # (Tu można dodać obsługę przednich/bocznych bumperów jeśli są w inputach)
        return None


class EncoderMonitor:
    """
    Zaawansowany monitoring enkoderów:
    - Wykrywanie poślizgu (slip)
    - Wykrywanie blokady (stall)
    - Korekcja toru jazdy
    - Detekcja nierówności podłoża
    """
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.expected_l = 0.0
        self.expected_r = 0.0
        self.actual_l = 0.0
        self.actual_r = 0.0

        # Historia dla detekcji trendów
        self.actual_history = deque(maxlen=20)
        self.expected_history = deque(maxlen=20)
        self.slip_history = deque(maxlen=20)

        # Liczniki
        self.slip_count = 0
        self.stall_count = 0
        self.total_slip = 0.0

        # Kalibracja
        self.pwm_deadzone = 20  # PWM poniżej tego nie rusza
        self.pwm_max_speed = 80  # PWM przy którym osiąga max speed

    def pwm_to_speed(self, pwm: float) -> float:
        """
        Przelicza PWM na oczekiwaną prędkość [m/s].
        Uwzględnia deadzone i nieliniowość silników.
        """
        pwm_abs = abs(pwm)
        if pwm_abs < self.pwm_deadzone:
            return 0.0

        sign = 1.0 if pwm > 0 else -1.0
        # Nieliniowa charakterystyka (silniki są nieliniowe)
        ratio = ((pwm_abs - self.pwm_deadzone) / self.pwm_max_speed) ** 0.8
        ratio = min(1.0, ratio)

        return sign * ratio * self.config.MAX_SPEED_MPS

    def set_expected(self, pwm_l: float, pwm_r: float):
        """Ustaw oczekiwane prędkości na podstawie PWM."""
        self.expected_l = self.pwm_to_speed(pwm_l)
        self.expected_r = self.pwm_to_speed(pwm_r)
        self.expected_history.append((self.expected_l, self.expected_r))

    def update(self, enc_l: float, enc_r: float, dt: float):
        """
        Aktualizuj rzeczywiste prędkości z enkoderów.
        enc_l/r powinny być w [m/s]
        """
        self.actual_l = enc_l
        self.actual_r = enc_r
        self.actual_history.append((enc_l, enc_r))

        # Oblicz poślizg dla każdego koła
        slip_l = self._calculate_slip(self.expected_l, enc_l)
        slip_r = self._calculate_slip(self.expected_r, enc_r)
        self.slip_history.append((slip_l, slip_r))

        # Aktualizuj liczniki
        if slip_l > 0.3 or slip_r > 0.3:  # 30% poślizgu
            self.slip_count += 1
            self.total_slip += (slip_l + slip_r) / 2.0
        else:
            self.slip_count = max(0, self.slip_count - 0.5)

        # Detekcja blokady (stall)
        if abs(self.expected_l) > 0.1 and abs(enc_l) < 0.01:
            self.stall_count += 1
        elif abs(self.expected_r) > 0.1 and abs(enc_r) < 0.01:
            self.stall_count += 1
        else:
            self.stall_count = max(0, self.stall_count - 0.5)

    def _calculate_slip(self, expected: float, actual: float) -> float:
        """
        Oblicza poślizg (0-1). 0 = brak, 1 = całkowity poślizg.
        """
        if abs(expected) < 0.01:  # Nie spodziewamy się ruchu
            return 0.0

        # Różnica względna
        if expected > 0:  # Jazda do przodu
            if actual < 0:  # Koło kręci się w tył? (bardzo zły poślizg)
                return 1.0
            slip = max(0, (expected - actual) / expected)
        else:  # Jazda do tyłu
            if actual > 0:
                return 1.0
            slip = max(0, (abs(expected) - abs(actual)) / abs(expected))

        return min(1.0, slip)

    def get_slip_ratio(self) -> float:
        """Średni poślizg w ostatnich N próbkach."""
        if not self.slip_history:
            return 0.0
        avg_slip_l = sum(s[0] for s in self.slip_history) / len(self.slip_history)
        avg_slip_r = sum(s[1] for s in self.slip_history) / len(self.slip_history)
        return (avg_slip_l + avg_slip_r) / 2.0

    def is_stalled(self) -> bool:
        """Czy któreś koło jest zablokowane?"""
        return self.stall_count > 5

    def is_slipping(self) -> bool:
        """Czy występuje znaczący poślizg?"""
        return self.get_slip_ratio() > 0.2

    def get_trajectory_correction(self) -> float:
        """
        Zwraca korekcję toru jazdy na podstawie różnicy enkoderów.
        Pozytywna wartość = skręć w prawo, negatywna = skręć w lewo.
        """
        if len(self.actual_history) < 5:
            return 0.0

        # Średnia różnica w ostatnich 5 próbkach
        recent = list(self.actual_history)[-5:]
        avg_diff = sum(r[0] - r[1] for r in recent) / len(recent)

        # Normalizuj do zakresu [-1, 1]
        max_diff = self.config.MAX_SPEED_MPS * 0.2
        correction = np.clip(avg_diff / max_diff, -1.0, 1.0)

        return correction

    def get_encoder_health(self) -> Dict:
        """Diagnostyka enkoderów."""
        return {
            'slip_ratio': self.get_slip_ratio(),
            'is_slipping': self.is_slipping(),
            'is_stalled': self.is_stalled(),
            'slip_count': self.slip_count,
            'stall_count': self.stall_count,
            'total_slip': self.total_slip,
            'correction': self.get_trajectory_correction(),
            'expected': (self.expected_l, self.expected_r),
            'actual': (self.actual_l, self.actual_r)
        }


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

    def update_odometry(self, encoder_ticks_l: float, encoder_ticks_r: float, dt: float):
        """
        Aktualizacja pozycji na podstawie rzeczywistych ticków enkoderów.
        A nie z PWM!
        """
        # Przelicz ticki na metry
        # Zakladamy ze encoder_ticks_l/r to PREDKOSC (m/s) lub przemieszczenie (m) w tym kroku?
        # Wczesniejsza implementacja uzywala vel_l, vel_r.
        # Jesli encoder_l to predkosc [m/s], to dist = vel * dt.

        # Ale instrukcja mowi: "encoder_ticks_l / ticks_per_metr".
        # To sugeruje ze input to ticki.
        # Ale loop() przyjmuje 'encoder_l: float' ktory jest uzywany jako avg_speed.
        # Wiec input do loop() to predkosc.
        # Zatem update_odometry powinno przyjmowac predkosc [m/s].

        # Jesli jednak chcemy uzyc ticks_per_m, to musimy znac ticki.
        # Zalozmy ze encoder_l to predkosc w m/s (juz przeliczona przez sterownik nizej).

        dist_l = encoder_ticks_l * dt
        dist_r = encoder_ticks_r * dt

        # Predkosc liniowa i katowa
        v = (encoder_ticks_l + encoder_ticks_r) / 2.0
        omega = (encoder_ticks_r - encoder_ticks_l) / self.config.WHEEL_BASE

        # Aktualizuj pozycje
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        self.theta += omega * dt

        # Normalizacja kata -pi do pi
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi


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
            # For DualLinearApproximator, weights are in q_weights/a_weights, so data.get('weights') might be None
            # But we can log something else.
            w_info = "weights_saved"
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

# =============================================================================
# SWARM CORE v5.5 FINAL
# =============================================================================

class SwarmCoreV55:
    """
    SWARM v5.5 - Production Final
    """

    def __init__(self):
        self.config = SwarmConfig()

        # Hardware
        self.lidar = LidarEngine(self.config)
        self.motors = DCMotorController(self.config)
        self.damper = VirtualDamper(self.config)

        # Nowe komponenty (Hierarchia)
        self.fusion = SensorFusion(self.config)
        self.bumper_system = BumperSystem(self.config)
        self.encoder_monitor = EncoderMonitor(self.config)
        self.stuck_detector = StuckDetector(self.config)
        self.movement_tracker = MovementTracker(self.config)

        # Feature Extractor (v5.9)
        self.feature_extractor = FeatureExtractor(self.config)

        # AI Layers
        self.brain = NeuralHybridBrain(self.config, self.feature_extractor)
        self.concept_graph = ConceptGraph(self.config)

        # Moduly
        self.lorenz = LorenzAttractor(self.config)
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
        # 2. Przygotuj dane do pickle
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
            'q_weights': self.brain.q_approx.q_weights.tolist(),
            'a_weights': self.brain.q_approx.a_weights.tolist(),
            'normalizer_state': self.brain.normalizer.get_state(),
            'epsilon': self.brain.epsilon,
            'lr': self.brain.lr,
            'concepts': concepts_data,
            'lorenz_state': self.lorenz.get_state(),
        }

        # 3. Zapisz do pliku pickle
        self.state_manager.save(data)



    def _load_state(self):
        saved = self.state_manager.load()
        if saved:
            norm_state = saved.get('normalizer_state')
            if norm_state and norm_state.get('n', 0) > 0:
                self.brain.normalizer.set_state(norm_state)
                logger.info(f"Loaded normalizer: n={norm_state['n']} samples")

            def load_weights(key, target_attr):
                if key in saved and saved[key] is not None:
                    loaded = np.array(saved[key])
                    current = getattr(self.brain.q_approx, target_attr)
                    if loaded.shape == current.shape:
                        setattr(self.brain.q_approx, target_attr, loaded)
                    elif loaded.shape[0] == current.shape[0] and loaded.shape[1] < current.shape[1]:
                        diff = current.shape[1] - loaded.shape[1]
                        logger.info(f"Padding {key}: {loaded.shape} -> {current.shape}")
                        padded = np.pad(loaded, ((0,0), (0,diff)), mode='constant', constant_values=0)
                        setattr(self.brain.q_approx, target_attr, padded)
                    else:
                        logger.warning(f"{key} shape mismatch: {loaded.shape} vs {current.shape}. Resetting.")

            load_weights('q_weights', 'q_weights')
            load_weights('a_weights', 'a_weights')

            self.brain.epsilon = saved.get('epsilon', self.config.EPSILON)
            self.brain.lr = saved.get('lr', self.config.LEARNING_RATE)
            self.brain.q_approx.set_learning_rate(self.brain.lr)

            saved_concepts = saved.get('concepts', {})
            if saved_concepts:
                for name, data in saved_concepts.items():
                    c = Concept(data['name'], data['sequence'])
                    c.activation = data['activation']
                    c.success_count = data['success_count']
                    c.usage_count = data['usage_count']
                    c.context = data.get('context', {})
                    self.concept_graph.concepts[name] = c

            lorenz_state = saved.get('lorenz_state')
            if lorenz_state:
                self.lorenz.x, self.lorenz.y, self.lorenz.z = lorenz_state
    def _compute_dynamic_safety(self, avg_speed: float) -> Tuple[float, float]:
        """
    def validate_safety_constraints(self, us_left_dist: float, us_right_dist: float,
                                   encoder_l: float, encoder_r: float,
                                   rear_bumper: int = 0) -> Optional[Tuple[Action, str]]:
        avg_speed = (encoder_l + encoder_r) / 2.0
        dyn_us, _dyn_lidar = self._compute_dynamic_safety(avg_speed)

        # --------------------------------------------------------------
        # LIDAR HARD SAFETY - TYLKO PONIŻEJ 10 cm!
        # --------------------------------------------------------------
        if self.lidar.min_dist < self.config.LIDAR_HARD_SAFETY_MIN:  # 10 cm
            # Anuluj wszystkie inne akcje
            self.rear_bumper_forward_remaining = 0
            self.stabilizer.force_unlock()

            # Wybierz kierunek ucieczki
            if us_left_dist > us_right_dist + 0.1:
                self.hard_reflex_action = Action.TURN_LEFT
            elif us_right_dist > us_left_dist + 0.1:
                self.hard_reflex_action = Action.TURN_RIGHT
            else:
                # Oba boki podobne - sprawdź tył
                rear_sectors = self.lidar.sectors_16[6:10]
                rear_blocked = float(np.mean(rear_sectors)) > 0.6
                if rear_blocked:
                    # Wszędzie źle - spin
                    self.hard_reflex_action = (Action.SPIN_LEFT if us_left_dist >= us_right_dist
                                              else Action.SPIN_RIGHT)
                else:
                    self.hard_reflex_action = Action.REVERSE

            self.hard_reflex_hold_remaining = self.config.HARD_REFLEX_HOLD_CYCLES
            return self.hard_reflex_action, "LIDAR_HARD_SAFETY"

        # --------------------------------------------------------------
        # US CHECK - TYLKO PONIŻEJ 10-15 cm (dynamiczne)
        # --------------------------------------------------------------
        us_front_min = min(us_left_dist, us_right_dist)

        # TYLKO gdy naprawdę blisko!
        if 0.01 < us_front_min < dyn_us:  # dyn_us to 8-15 cm
            # Sprawdź LIDAR czy potwierdza
            if self.config.REVERSE_LIDAR_CHECK:
                front_blocked = self.lidar.check_front_sectors_blocked(
                    threshold=0.3,  # Wyższy próg - musi być naprawdę blisko
                    num_sectors=self.config.REVERSE_LIDAR_SECTORS
                )
                if not front_blocked:
                    # LIDAR mówi że jest miejsce - NIE COFAJ!
                    return None, "US_WARNING_ONLY"

            # Faktycznie jest blisko - delikatnie cofnij
            self.stabilizer.force_unlock()
            self.hard_reflex_action = Action.REVERSE
            self.hard_reflex_hold_remaining = 1  # Tylko 1 cykl!
            return Action.REVERSE, "CLOSE_CALL"

        return None
        Oblicza dynamiczne progi bezpieczeństwa.
        Teraz znacznie odważniejsze - reaguje tylko naprawdę blisko!
    def validate_safety_constraints(self, us_left_dist: float, us_right_dist: float,
                                   encoder_l: float, encoder_r: float,
                                   rear_bumper: int = 0) -> Optional[Tuple[Action, str]]:
        avg_speed = (encoder_l + encoder_r) / 2.0
        dyn_us, _dyn_lidar = self._compute_dynamic_safety(avg_speed)

        # --------------------------------------------------------------
        # LIDAR HARD SAFETY - TYLKO PONIŻEJ 10 cm!
        # --------------------------------------------------------------
        if self.lidar.min_dist < self.config.LIDAR_HARD_SAFETY_MIN:  # 10 cm
            # Anuluj wszystkie inne akcje
            self.rear_bumper_forward_remaining = 0
            self.stabilizer.force_unlock()

            # Wybierz kierunek ucieczki
            if us_left_dist > us_right_dist + 0.1:
                self.hard_reflex_action = Action.TURN_LEFT
            elif us_right_dist > us_left_dist + 0.1:
                self.hard_reflex_action = Action.TURN_RIGHT
            else:
                # Oba boki podobne - sprawdź tył
                rear_sectors = self.lidar.sectors_16[6:10]
                rear_blocked = float(np.mean(rear_sectors)) > 0.6
                if rear_blocked:
                    # Wszędzie źle - spin
                    self.hard_reflex_action = (Action.SPIN_LEFT if us_left_dist >= us_right_dist
                                              else Action.SPIN_RIGHT)
                else:
                    self.hard_reflex_action = Action.REVERSE

            self.hard_reflex_hold_remaining = self.config.HARD_REFLEX_HOLD_CYCLES
            return self.hard_reflex_action, "LIDAR_HARD_SAFETY"

        # --------------------------------------------------------------
        # US CHECK - TYLKO PONIŻEJ 10-15 cm (dynamiczne)
        # --------------------------------------------------------------
        us_front_min = min(us_left_dist, us_right_dist)

        # TYLKO gdy naprawdę blisko!
        if 0.01 < us_front_min < dyn_us:  # dyn_us to 8-15 cm
            # Sprawdź LIDAR czy potwierdza
            if self.config.REVERSE_LIDAR_CHECK:
                front_blocked = self.lidar.check_front_sectors_blocked(
                    threshold=0.3,  # Wyższy próg - musi być naprawdę blisko
                    num_sectors=self.config.REVERSE_LIDAR_SECTORS
                )
                if not front_blocked:
                    # LIDAR mówi że jest miejsce - NIE COFAJ!
                    return None, "US_WARNING_ONLY"

            # Faktycznie jest blisko - delikatnie cofnij
            self.stabilizer.force_unlock()
            self.hard_reflex_action = Action.REVERSE
            self.hard_reflex_hold_remaining = 1  # Tylko 1 cykl!
            return Action.REVERSE, "CLOSE_CALL"

        return None
        """
        scale = self.config.SAFETY_DIST_SPEED_SCALE
    def validate_safety_constraints(self, us_left_dist: float, us_right_dist: float,
                                   encoder_l: float, encoder_r: float,
                                   rear_bumper: int = 0) -> Optional[Tuple[Action, str]]:
        avg_speed = (encoder_l + encoder_r) / 2.0
        dyn_us, _dyn_lidar = self._compute_dynamic_safety(avg_speed)

        # --------------------------------------------------------------
        # LIDAR HARD SAFETY - TYLKO PONIŻEJ 10 cm!
        # --------------------------------------------------------------
        if self.lidar.min_dist < self.config.LIDAR_HARD_SAFETY_MIN:  # 10 cm
            # Anuluj wszystkie inne akcje
            self.rear_bumper_forward_remaining = 0
            self.stabilizer.force_unlock()

            # Wybierz kierunek ucieczki
            if us_left_dist > us_right_dist + 0.1:
                self.hard_reflex_action = Action.TURN_LEFT
            elif us_right_dist > us_left_dist + 0.1:
                self.hard_reflex_action = Action.TURN_RIGHT
            else:
                # Oba boki podobne - sprawdź tył
                rear_sectors = self.lidar.sectors_16[6:10]
                rear_blocked = float(np.mean(rear_sectors)) > 0.6
                if rear_blocked:
                    # Wszędzie źle - spin
                    self.hard_reflex_action = (Action.SPIN_LEFT if us_left_dist >= us_right_dist
                                              else Action.SPIN_RIGHT)
                else:
                    self.hard_reflex_action = Action.REVERSE

            self.hard_reflex_hold_remaining = self.config.HARD_REFLEX_HOLD_CYCLES
            return self.hard_reflex_action, "LIDAR_HARD_SAFETY"

        # --------------------------------------------------------------
        # US CHECK - TYLKO PONIŻEJ 10-15 cm (dynamiczne)
        # --------------------------------------------------------------
        us_front_min = min(us_left_dist, us_right_dist)

        # TYLKO gdy naprawdę blisko!
        if 0.01 < us_front_min < dyn_us:  # dyn_us to 8-15 cm
            # Sprawdź LIDAR czy potwierdza
            if self.config.REVERSE_LIDAR_CHECK:
                front_blocked = self.lidar.check_front_sectors_blocked(
                    threshold=0.3,  # Wyższy próg - musi być naprawdę blisko
                    num_sectors=self.config.REVERSE_LIDAR_SECTORS
                )
                if not front_blocked:
                    # LIDAR mówi że jest miejsce - NIE COFAJ!
                    return None, "US_WARNING_ONLY"

            # Faktycznie jest blisko - delikatnie cofnij
            self.stabilizer.force_unlock()
            self.hard_reflex_action = Action.REVERSE
            self.hard_reflex_hold_remaining = 1  # Tylko 1 cykl!
            return Action.REVERSE, "CLOSE_CALL"

        return None
        v = abs(avg_speed)

    def validate_safety_constraints(self, us_left_dist: float, us_right_dist: float,
                                   encoder_l: float, encoder_r: float,
                                   rear_bumper: int = 0) -> Optional[Tuple[Action, str]]:
        avg_speed = (encoder_l + encoder_r) / 2.0
        dyn_us, _dyn_lidar = self._compute_dynamic_safety(avg_speed)

        # --------------------------------------------------------------
        # LIDAR HARD SAFETY - TYLKO PONIŻEJ 10 cm!
        # --------------------------------------------------------------
        if self.lidar.min_dist < self.config.LIDAR_HARD_SAFETY_MIN:  # 10 cm
            # Anuluj wszystkie inne akcje
            self.rear_bumper_forward_remaining = 0
            self.stabilizer.force_unlock()

            # Wybierz kierunek ucieczki
            if us_left_dist > us_right_dist + 0.1:
                self.hard_reflex_action = Action.TURN_LEFT
            elif us_right_dist > us_left_dist + 0.1:
                self.hard_reflex_action = Action.TURN_RIGHT
            else:
                # Oba boki podobne - sprawdź tył
                rear_sectors = self.lidar.sectors_16[6:10]
                rear_blocked = float(np.mean(rear_sectors)) > 0.6
                if rear_blocked:
                    # Wszędzie źle - spin
                    self.hard_reflex_action = (Action.SPIN_LEFT if us_left_dist >= us_right_dist
                                              else Action.SPIN_RIGHT)
                else:
                    self.hard_reflex_action = Action.REVERSE

            self.hard_reflex_hold_remaining = self.config.HARD_REFLEX_HOLD_CYCLES
            return self.hard_reflex_action, "LIDAR_HARD_SAFETY"

        # --------------------------------------------------------------
        # US CHECK - TYLKO PONIŻEJ 10-15 cm (dynamiczne)
        # --------------------------------------------------------------
        us_front_min = min(us_left_dist, us_right_dist)

        # TYLKO gdy naprawdę blisko!
        if 0.01 < us_front_min < dyn_us:  # dyn_us to 8-15 cm
            # Sprawdź LIDAR czy potwierdza
            if self.config.REVERSE_LIDAR_CHECK:
                front_blocked = self.lidar.check_front_sectors_blocked(
                    threshold=0.3,  # Wyższy próg - musi być naprawdę blisko
                    num_sectors=self.config.REVERSE_LIDAR_SECTORS
                )
                if not front_blocked:
                    # LIDAR mówi że jest miejsce - NIE COFAJ!
                    return None, "US_WARNING_ONLY"

            # Faktycznie jest blisko - delikatnie cofnij
            self.stabilizer.force_unlock()
            self.hard_reflex_action = Action.REVERSE
            self.hard_reflex_hold_remaining = 1  # Tylko 1 cykl!
            return Action.REVERSE, "CLOSE_CALL"

        return None
        # US safety - od 8 cm do 15 cm zależnie od prędkości
        us_safety = self.config.US_SAFETY_DIST + (scale * v)
    def validate_safety_constraints(self, us_left_dist: float, us_right_dist: float,
                                   encoder_l: float, encoder_r: float,
                                   rear_bumper: int = 0) -> Optional[Tuple[Action, str]]:
        avg_speed = (encoder_l + encoder_r) / 2.0
        dyn_us, _dyn_lidar = self._compute_dynamic_safety(avg_speed)

        # --------------------------------------------------------------
        # LIDAR HARD SAFETY - TYLKO PONIŻEJ 10 cm!
        # --------------------------------------------------------------
        if self.lidar.min_dist < self.config.LIDAR_HARD_SAFETY_MIN:  # 10 cm
            # Anuluj wszystkie inne akcje
            self.rear_bumper_forward_remaining = 0
            self.stabilizer.force_unlock()

            # Wybierz kierunek ucieczki
            if us_left_dist > us_right_dist + 0.1:
                self.hard_reflex_action = Action.TURN_LEFT
            elif us_right_dist > us_left_dist + 0.1:
                self.hard_reflex_action = Action.TURN_RIGHT
            else:
                # Oba boki podobne - sprawdź tył
                rear_sectors = self.lidar.sectors_16[6:10]
                rear_blocked = float(np.mean(rear_sectors)) > 0.6
                if rear_blocked:
                    # Wszędzie źle - spin
                    self.hard_reflex_action = (Action.SPIN_LEFT if us_left_dist >= us_right_dist
                                              else Action.SPIN_RIGHT)
                else:
                    self.hard_reflex_action = Action.REVERSE

            self.hard_reflex_hold_remaining = self.config.HARD_REFLEX_HOLD_CYCLES
            return self.hard_reflex_action, "LIDAR_HARD_SAFETY"

        # --------------------------------------------------------------
        # US CHECK - TYLKO PONIŻEJ 10-15 cm (dynamiczne)
        # --------------------------------------------------------------
        us_front_min = min(us_left_dist, us_right_dist)

        # TYLKO gdy naprawdę blisko!
        if 0.01 < us_front_min < dyn_us:  # dyn_us to 8-15 cm
            # Sprawdź LIDAR czy potwierdza
            if self.config.REVERSE_LIDAR_CHECK:
                front_blocked = self.lidar.check_front_sectors_blocked(
                    threshold=0.3,  # Wyższy próg - musi być naprawdę blisko
                    num_sectors=self.config.REVERSE_LIDAR_SECTORS
                )
                if not front_blocked:
                    # LIDAR mówi że jest miejsce - NIE COFAJ!
                    return None, "US_WARNING_ONLY"

            # Faktycznie jest blisko - delikatnie cofnij
            self.stabilizer.force_unlock()
            self.hard_reflex_action = Action.REVERSE
            self.hard_reflex_hold_remaining = 1  # Tylko 1 cykl!
            return Action.REVERSE, "CLOSE_CALL"

        return None
        us_safety = max(self.config.SAFETY_US_MIN,
                        min(self.config.SAFETY_US_MAX, us_safety))
    def validate_safety_constraints(self, us_left_dist: float, us_right_dist: float,
                                   encoder_l: float, encoder_r: float,
                                   rear_bumper: int = 0) -> Optional[Tuple[Action, str]]:
        avg_speed = (encoder_l + encoder_r) / 2.0
        dyn_us, _dyn_lidar = self._compute_dynamic_safety(avg_speed)

        # --------------------------------------------------------------
        # LIDAR HARD SAFETY - TYLKO PONIŻEJ 10 cm!
        # --------------------------------------------------------------
        if self.lidar.min_dist < self.config.LIDAR_HARD_SAFETY_MIN:  # 10 cm
            # Anuluj wszystkie inne akcje
            self.rear_bumper_forward_remaining = 0
            self.stabilizer.force_unlock()

            # Wybierz kierunek ucieczki
            if us_left_dist > us_right_dist + 0.1:
                self.hard_reflex_action = Action.TURN_LEFT
            elif us_right_dist > us_left_dist + 0.1:
                self.hard_reflex_action = Action.TURN_RIGHT
            else:
                # Oba boki podobne - sprawdź tył
                rear_sectors = self.lidar.sectors_16[6:10]
                rear_blocked = float(np.mean(rear_sectors)) > 0.6
                if rear_blocked:
                    # Wszędzie źle - spin
                    self.hard_reflex_action = (Action.SPIN_LEFT if us_left_dist >= us_right_dist
                                              else Action.SPIN_RIGHT)
                else:
                    self.hard_reflex_action = Action.REVERSE

            self.hard_reflex_hold_remaining = self.config.HARD_REFLEX_HOLD_CYCLES
            return self.hard_reflex_action, "LIDAR_HARD_SAFETY"

        # --------------------------------------------------------------
        # US CHECK - TYLKO PONIŻEJ 10-15 cm (dynamiczne)
        # --------------------------------------------------------------
        us_front_min = min(us_left_dist, us_right_dist)

        # TYLKO gdy naprawdę blisko!
        if 0.01 < us_front_min < dyn_us:  # dyn_us to 8-15 cm
            # Sprawdź LIDAR czy potwierdza
            if self.config.REVERSE_LIDAR_CHECK:
                front_blocked = self.lidar.check_front_sectors_blocked(
                    threshold=0.3,  # Wyższy próg - musi być naprawdę blisko
                    num_sectors=self.config.REVERSE_LIDAR_SECTORS
                )
                if not front_blocked:
                    # LIDAR mówi że jest miejsce - NIE COFAJ!
                    return None, "US_WARNING_ONLY"

            # Faktycznie jest blisko - delikatnie cofnij
            self.stabilizer.force_unlock()
            self.hard_reflex_action = Action.REVERSE
            self.hard_reflex_hold_remaining = 1  # Tylko 1 cykl!
            return Action.REVERSE, "CLOSE_CALL"

        return None

        # LIDAR safety - analogicznie
    def validate_safety_constraints(self, us_left_dist: float, us_right_dist: float,
                                   encoder_l: float, encoder_r: float,
                                   rear_bumper: int = 0) -> Optional[Tuple[Action, str]]:
        avg_speed = (encoder_l + encoder_r) / 2.0
        dyn_us, _dyn_lidar = self._compute_dynamic_safety(avg_speed)

        # --------------------------------------------------------------
        # LIDAR HARD SAFETY - TYLKO PONIŻEJ 10 cm!
        # --------------------------------------------------------------
        if self.lidar.min_dist < self.config.LIDAR_HARD_SAFETY_MIN:  # 10 cm
            # Anuluj wszystkie inne akcje
            self.rear_bumper_forward_remaining = 0
            self.stabilizer.force_unlock()

            # Wybierz kierunek ucieczki
            if us_left_dist > us_right_dist + 0.1:
                self.hard_reflex_action = Action.TURN_LEFT
            elif us_right_dist > us_left_dist + 0.1:
                self.hard_reflex_action = Action.TURN_RIGHT
            else:
                # Oba boki podobne - sprawdź tył
                rear_sectors = self.lidar.sectors_16[6:10]
                rear_blocked = float(np.mean(rear_sectors)) > 0.6
                if rear_blocked:
                    # Wszędzie źle - spin
                    self.hard_reflex_action = (Action.SPIN_LEFT if us_left_dist >= us_right_dist
                                              else Action.SPIN_RIGHT)
                else:
                    self.hard_reflex_action = Action.REVERSE

            self.hard_reflex_hold_remaining = self.config.HARD_REFLEX_HOLD_CYCLES
            return self.hard_reflex_action, "LIDAR_HARD_SAFETY"

        # --------------------------------------------------------------
        # US CHECK - TYLKO PONIŻEJ 10-15 cm (dynamiczne)
        # --------------------------------------------------------------
        us_front_min = min(us_left_dist, us_right_dist)

        # TYLKO gdy naprawdę blisko!
        if 0.01 < us_front_min < dyn_us:  # dyn_us to 8-15 cm
            # Sprawdź LIDAR czy potwierdza
            if self.config.REVERSE_LIDAR_CHECK:
                front_blocked = self.lidar.check_front_sectors_blocked(
                    threshold=0.3,  # Wyższy próg - musi być naprawdę blisko
                    num_sectors=self.config.REVERSE_LIDAR_SECTORS
                )
                if not front_blocked:
                    # LIDAR mówi że jest miejsce - NIE COFAJ!
                    return None, "US_WARNING_ONLY"

            # Faktycznie jest blisko - delikatnie cofnij
            self.stabilizer.force_unlock()
            self.hard_reflex_action = Action.REVERSE
            self.hard_reflex_hold_remaining = 1  # Tylko 1 cykl!
            return Action.REVERSE, "CLOSE_CALL"

        return None
    def validate_safety_constraints(self, us_left_dist: float, us_right_dist: float,
                                   encoder_l: float, encoder_r: float,
                                   rear_bumper: int = 0) -> Optional[Tuple[Action, str]]:
        avg_speed = (encoder_l + encoder_r) / 2.0
        dyn_us, _dyn_lidar = self._compute_dynamic_safety(avg_speed)

        # --------------------------------------------------------------
        # LIDAR HARD SAFETY - TYLKO PONIŻEJ 10 cm!
        # --------------------------------------------------------------
        if self.lidar.min_dist < self.config.LIDAR_HARD_SAFETY_MIN:  # 10 cm
            # Anuluj wszystkie inne akcje
            self.rear_bumper_forward_remaining = 0
            self.stabilizer.force_unlock()

            # Wybierz kierunek ucieczki
            if us_left_dist > us_right_dist + 0.1:
                self.hard_reflex_action = Action.TURN_LEFT
            elif us_right_dist > us_left_dist + 0.1:
                self.hard_reflex_action = Action.TURN_RIGHT
            else:
                # Oba boki podobne - sprawdź tył
                rear_sectors = self.lidar.sectors_16[6:10]
                rear_blocked = float(np.mean(rear_sectors)) > 0.6
                if rear_blocked:
                    # Wszędzie źle - spin
                    self.hard_reflex_action = (Action.SPIN_LEFT if us_left_dist >= us_right_dist
                                              else Action.SPIN_RIGHT)
                else:
                    self.hard_reflex_action = Action.REVERSE

            self.hard_reflex_hold_remaining = self.config.HARD_REFLEX_HOLD_CYCLES
            return self.hard_reflex_action, "LIDAR_HARD_SAFETY"

        # --------------------------------------------------------------
        # US CHECK - TYLKO PONIŻEJ 10-15 cm (dynamiczne)
        # --------------------------------------------------------------
        us_front_min = min(us_left_dist, us_right_dist)

        # TYLKO gdy naprawdę blisko!
        if 0.01 < us_front_min < dyn_us:  # dyn_us to 8-15 cm
            # Sprawdź LIDAR czy potwierdza
            if self.config.REVERSE_LIDAR_CHECK:
                front_blocked = self.lidar.check_front_sectors_blocked(
                    threshold=0.3,  # Wyższy próg - musi być naprawdę blisko
                    num_sectors=self.config.REVERSE_LIDAR_SECTORS
                )
                if not front_blocked:
                    # LIDAR mówi że jest miejsce - NIE COFAJ!
                    return None, "US_WARNING_ONLY"

            # Faktycznie jest blisko - delikatnie cofnij
            self.stabilizer.force_unlock()
            self.hard_reflex_action = Action.REVERSE
            self.hard_reflex_hold_remaining = 1  # Tylko 1 cykl!
            return Action.REVERSE, "CLOSE_CALL"

        return None

    def validate_safety_constraints(self, us_left_dist: float, us_right_dist: float,
                                   encoder_l: float, encoder_r: float,
                                   rear_bumper: int = 0) -> Optional[Tuple[Action, str]]:
        avg_speed = (encoder_l + encoder_r) / 2.0
        dyn_us, _dyn_lidar = self._compute_dynamic_safety(avg_speed)

        # --------------------------------------------------------------
        # LIDAR HARD SAFETY - TYLKO PONIŻEJ 10 cm!
        # --------------------------------------------------------------
        if self.lidar.min_dist < self.config.LIDAR_HARD_SAFETY_MIN:  # 10 cm
            # Anuluj wszystkie inne akcje
            self.rear_bumper_forward_remaining = 0
            self.stabilizer.force_unlock()

            # Wybierz kierunek ucieczki
            if us_left_dist > us_right_dist + 0.1:
                self.hard_reflex_action = Action.TURN_LEFT
            elif us_right_dist > us_left_dist + 0.1:
                self.hard_reflex_action = Action.TURN_RIGHT
            else:
                # Oba boki podobne - sprawdź tył
                rear_sectors = self.lidar.sectors_16[6:10]
                rear_blocked = float(np.mean(rear_sectors)) > 0.6
                if rear_blocked:
                    # Wszędzie źle - spin
                    self.hard_reflex_action = (Action.SPIN_LEFT if us_left_dist >= us_right_dist
                                              else Action.SPIN_RIGHT)
                else:
                    self.hard_reflex_action = Action.REVERSE

            self.hard_reflex_hold_remaining = self.config.HARD_REFLEX_HOLD_CYCLES
            return self.hard_reflex_action, "LIDAR_HARD_SAFETY"

        # --------------------------------------------------------------
        # US CHECK - TYLKO PONIŻEJ 10-15 cm (dynamiczne)
        # --------------------------------------------------------------
        us_front_min = min(us_left_dist, us_right_dist)

        # TYLKO gdy naprawdę blisko!
        if 0.01 < us_front_min < dyn_us:  # dyn_us to 8-15 cm
            # Sprawdź LIDAR czy potwierdza
            if self.config.REVERSE_LIDAR_CHECK:
                front_blocked = self.lidar.check_front_sectors_blocked(
                    threshold=0.3,  # Wyższy próg - musi być naprawdę blisko
                    num_sectors=self.config.REVERSE_LIDAR_SECTORS
                )
                if not front_blocked:
                    # LIDAR mówi że jest miejsce - NIE COFAJ!
                    return None, "US_WARNING_ONLY"

            # Faktycznie jest blisko - delikatnie cofnij
            self.stabilizer.force_unlock()
            self.hard_reflex_action = Action.REVERSE
            self.hard_reflex_hold_remaining = 1  # Tylko 1 cykl!
            return Action.REVERSE, "CLOSE_CALL"

        return None
        return us_safety, lidar_safety
    def validate_safety_constraints(self, us_left_dist: float, us_right_dist: float,
                                   encoder_l: float, encoder_r: float,
                                   rear_bumper: int = 0) -> Optional[Tuple[Action, str]]:
        avg_speed = (encoder_l + encoder_r) / 2.0
        dyn_us, _dyn_lidar = self._compute_dynamic_safety(avg_speed)

        # --------------------------------------------------------------
        # LIDAR HARD SAFETY - TYLKO PONIŻEJ 10 cm!
        # --------------------------------------------------------------
        if self.lidar.min_dist < self.config.LIDAR_HARD_SAFETY_MIN:  # 10 cm
            # Anuluj wszystkie inne akcje
            self.rear_bumper_forward_remaining = 0
            self.stabilizer.force_unlock()

            # Wybierz kierunek ucieczki
            if us_left_dist > us_right_dist + 0.1:
                self.hard_reflex_action = Action.TURN_LEFT
            elif us_right_dist > us_left_dist + 0.1:
                self.hard_reflex_action = Action.TURN_RIGHT
            else:
                # Oba boki podobne - sprawdź tył
                rear_sectors = self.lidar.sectors_16[6:10]
                rear_blocked = float(np.mean(rear_sectors)) > 0.6
                if rear_blocked:
                    # Wszędzie źle - spin
                    self.hard_reflex_action = (Action.SPIN_LEFT if us_left_dist >= us_right_dist
                                              else Action.SPIN_RIGHT)
                else:
                    self.hard_reflex_action = Action.REVERSE

            self.hard_reflex_hold_remaining = self.config.HARD_REFLEX_HOLD_CYCLES
            return self.hard_reflex_action, "LIDAR_HARD_SAFETY"

        # --------------------------------------------------------------
        # US CHECK - TYLKO PONIŻEJ 10-15 cm (dynamiczne)
        # --------------------------------------------------------------
        us_front_min = min(us_left_dist, us_right_dist)

        # TYLKO gdy naprawdę blisko!
        if 0.01 < us_front_min < dyn_us:  # dyn_us to 8-15 cm
            # Sprawdź LIDAR czy potwierdza
            if self.config.REVERSE_LIDAR_CHECK:
                front_blocked = self.lidar.check_front_sectors_blocked(
                    threshold=0.3,  # Wyższy próg - musi być naprawdę blisko
                    num_sectors=self.config.REVERSE_LIDAR_SECTORS
                )
                if not front_blocked:
                    # LIDAR mówi że jest miejsce - NIE COFAJ!
                    return None, "US_WARNING_ONLY"

            # Faktycznie jest blisko - delikatnie cofnij
            self.stabilizer.force_unlock()
            self.hard_reflex_action = Action.REVERSE
            self.hard_reflex_hold_remaining = 1  # Tylko 1 cykl!
            return Action.REVERSE, "CLOSE_CALL"

        return None

    def validate_safety_constraints(self, us_left_dist: float, us_right_dist: float,
                                   encoder_l: float, encoder_r: float,
                                   rear_bumper: int = 0) -> Optional[Tuple[Action, str]]:
        avg_speed = (encoder_l + encoder_r) / 2.0
        dyn_us, _dyn_lidar = self._compute_dynamic_safety(avg_speed)

        # --------------------------------------------------------------
        # LIDAR HARD SAFETY - TYLKO PONIŻEJ 10 cm!
        # --------------------------------------------------------------
        if self.lidar.min_dist < self.config.LIDAR_HARD_SAFETY_MIN:  # 10 cm
            # Anuluj wszystkie inne akcje
            self.rear_bumper_forward_remaining = 0
            self.stabilizer.force_unlock()

            # Wybierz kierunek ucieczki
            if us_left_dist > us_right_dist + 0.1:
                self.hard_reflex_action = Action.TURN_LEFT
            elif us_right_dist > us_left_dist + 0.1:
                self.hard_reflex_action = Action.TURN_RIGHT
            else:
                # Oba boki podobne - sprawdź tył
                rear_sectors = self.lidar.sectors_16[6:10]
                rear_blocked = float(np.mean(rear_sectors)) > 0.6
                if rear_blocked:
                    # Wszędzie źle - spin
                    self.hard_reflex_action = (Action.SPIN_LEFT if us_left_dist >= us_right_dist
                                              else Action.SPIN_RIGHT)
                else:
                    self.hard_reflex_action = Action.REVERSE

            self.hard_reflex_hold_remaining = self.config.HARD_REFLEX_HOLD_CYCLES
            return self.hard_reflex_action, "LIDAR_HARD_SAFETY"

        # --------------------------------------------------------------
        # US CHECK - TYLKO PONIŻEJ 10-15 cm (dynamiczne)
        # --------------------------------------------------------------
        us_front_min = min(us_left_dist, us_right_dist)

        # TYLKO gdy naprawdę blisko!
        if 0.01 < us_front_min < dyn_us:  # dyn_us to 8-15 cm
            # Sprawdź LIDAR czy potwierdza
            if self.config.REVERSE_LIDAR_CHECK:
                front_blocked = self.lidar.check_front_sectors_blocked(
                    threshold=0.3,  # Wyższy próg - musi być naprawdę blisko
                    num_sectors=self.config.REVERSE_LIDAR_SECTORS
                )
                if not front_blocked:
                    # LIDAR mówi że jest miejsce - NIE COFAJ!
                    return None, "US_WARNING_ONLY"

            # Faktycznie jest blisko - delikatnie cofnij
            self.stabilizer.force_unlock()
            self.hard_reflex_action = Action.REVERSE
            self.hard_reflex_hold_remaining = 1  # Tylko 1 cykl!
            return Action.REVERSE, "CLOSE_CALL"

        return None

    def _decide_from_fusion(self, enc_l, enc_r) -> Tuple[Optional[Action], str]:
        # POZIOM 1: Damper handled in loop
        # POZIOM 2: Blokada
        if self.stuck_detector.update(self.fusion.front_dist, enc_l, enc_r):
            logger.warning("STUCK DETECTED! Uwalniam robota...")
            if self.fusion.rear_dist > 0.3:
                return Action.REVERSE, "STUCK_REVERSE"
            else:
                if self.fusion.left_dist >= self.fusion.right_dist:
                    return Action.SPIN_LEFT, "STUCK_SPIN_LEFT"
                else:
                    return Action.SPIN_RIGHT, "STUCK_SPIN_RIGHT"

        # POZIOM 3: Bezpieczenstwo
        if self.fusion.front_dist < self.config.LIDAR_HARD_SAFETY_MIN:
             if self.fusion.rear_dist > 0.3:
                  return Action.REVERSE, "SAFETY_REVERSE"
             else:
                  if self.fusion.left_dist >= self.fusion.right_dist:
                       return Action.SPIN_LEFT, "SAFETY_SPIN"
                  else:
                       return Action.SPIN_RIGHT, "SAFETY_SPIN"
        return None, "NORMAL"

        # POZIOM 1: Damper handled in loop
        # POZIOM 2: Blokada
        if self.stuck_detector.update(self.fusion.front_dist, enc_l, enc_r):
            logger.warning("STUCK DETECTED! Uwalniam robota...")
            if self.fusion.rear_dist > 0.3:
                return Action.REVERSE, "STUCK_REVERSE"
            else:
                if self.fusion.left_dist >= self.fusion.right_dist:
                    return Action.SPIN_LEFT, "STUCK_SPIN_LEFT"
                else:
                    return Action.SPIN_RIGHT, "STUCK_SPIN_RIGHT"

        # POZIOM 3: Bezpieczenstwo
        if self.fusion.front_dist < self.config.LIDAR_HARD_SAFETY_MIN:
             if self.fusion.rear_dist > 0.3:
                  return Action.REVERSE, "SAFETY_REVERSE"
             else:
                  if self.fusion.left_dist >= self.fusion.right_dist:
                       return Action.SPIN_LEFT, "SAFETY_SPIN"
                  else:
                       return Action.SPIN_RIGHT, "SAFETY_SPIN"
        return None, "NORMAL"

    def loop(self, lidar_points: List[Tuple[float, float]],
             encoder_l: float, encoder_r: float,
             motor_current: float,
             us_left_dist: float = 3.0, us_right_dist: float = 3.0,
             rear_bumper: int = 0,
             dt: float = 0.033) -> Tuple[float, float]:

        self.cycle_count += 1

        # 0. Update Encoder Monitor (Observe effect of previous action)
        self.encoder_monitor.update(encoder_l, encoder_r, dt)
        enc_stats = self.encoder_monitor.get_encoder_health()

        lidar_16 = self.lidar.process(lidar_points)
        self.fusion.update(us_left_dist, us_right_dist, lidar_16)

        avg_speed = (encoder_l + encoder_r) / 2.0

        bumper_action = self.bumper_system.check_collision(rear_bumper)
        if bumper_action:
             final_action = bumper_action
             source = "BUMPER_PRIORITY"
             self.stabilizer.force_unlock()
        else:
             override_action, override_source = self._decide_from_fusion(encoder_l, encoder_r)
             if override_action:
                  final_action = override_action
                  source = override_source
                  self.stabilizer.force_unlock()
             else:
                  self.lorenz.step()
                  free_angle, free_mag = self.instinct.compute_free_space_vector(lidar_16)
                  front_occ = float(np.mean([lidar_16[14], lidar_16[15], lidar_16[0], lidar_16[1]]))
                  front_clearance = 1.0 - front_occ

                  instinct_bias = self.instinct.get_bias_for_action(
                      free_angle, magnitude=free_mag, front_clearance=front_clearance,
                      us_left=us_left_dist, us_right=us_right_dist
                  )
                  instinct_bias = self.instinct.apply_us_bias(instinct_bias, us_left_dist, us_right_dist)

                  features = self.brain.get_features(
                      lidar_16, us_left_dist, us_right_dist,
                      encoder_l, encoder_r,
                      self.lorenz.x_norm, self.lorenz.z_norm,
                      rear_bumper, self.lidar.min_dist,
                      last_action=self._last_action_type,
                      free_angle=free_angle, free_mag=free_mag,
                      slip_ratio=enc_stats['slip_ratio'],
                      is_stalled=enc_stats['is_stalled'],
                      traj_correction=enc_stats['correction'],
                      stall_intensity=enc_stats['stall_count'] / 10.0
                  )

                  context = {'min_dist': self.lidar.min_dist, 'us_left': us_left_dist, 'us_right': us_right_dist}
                  best_concept = self.concept_graph.get_best_concept(context)
                  concept_suggestion = None
                  if best_concept:
                      concept_suggestion = self.concept_graph.get_next_action_from_concept(best_concept)
                  # Loguj koncepty TYLKO co 50 cykli i gdy się zmieniają
                  if self.cycle_count % 50 == 0:
                      if not hasattr(self, "_last_concept") or self._last_concept != best_concept.name:
                          self._last_concept = best_concept.name
                          logger.info(
                              f"[CONCEPT] Uzyto konceptu '{best_concept.name}' "
                              f"(aktywacja={best_concept.activation:.2f}) "
                              f"-> sugestia: {concept_suggestion.name if concept_suggestion else 'None'}"
                          )

                  forced_turn = self.anti_stagnation.should_force_turn()
                  if forced_turn is not None:
                      final_action = forced_turn
                      source = "STAGNATION_FORCE"
                      self.stabilizer.force_unlock()
                  else:
                      action_candidate, source, _ = self.brain.decide(features, instinct_bias, concept_suggestion)
                      final_action = self.stabilizer.update(action_candidate)

        if final_action == self._last_action_type:
            self._action_repeat_count += 1
        else:
            self._action_repeat_count = 0
            self._last_action_type = final_action

        self.motors.update_odometry(encoder_l, encoder_r, dt)
        self.movement_tracker.update(final_action, self.motors.x, self.motors.y, self.cycle_count)

        if self.movement_tracker.is_oscillating() and source not in ["BUMPER_PRIORITY", "STUCK_REVERSE", "STUCK_SPIN_LEFT", "STUCK_SPIN_RIGHT"]:
            if self.fusion.left_dist >= self.fusion.right_dist:
                final_action = Action.SPIN_LEFT
            else:
                final_action = Action.SPIN_RIGHT
            source = "ANTI_OSCILLATION"
            self.stabilizer.force_unlock()

        reward = self.damper.compute_reward(encoder_l, encoder_r, motor_current, final_action)
        # NOWE: KARA za niepotrzebne cofanie!
        if final_action == Action.REVERSE and self.lidar.min_dist > 0.20:  # >20 cm
            unnecessary_penalty = -2.0 * (self.lidar.min_dist / 0.5)  # Kara proporcjonalna
            reward += unnecessary_penalty
            if self.cycle_count % 50 == 0:
                logger.info(f"⚠️ Niepotrzebne cofanie! dist={self.lidar.min_dist:.2f}m kara={unnecessary_penalty:.1f}")

        # NAGRODA za odważną jazdę do przodu
        if final_action == Action.FORWARD and self.lidar.min_dist < 0.30:  # 20-30 cm
            bravery_bonus = 1.0 * (1.0 - self.lidar.min_dist/0.3)  # Im bliżej, tym więcej
            reward += bravery_bonus

        # Penalize stall
        if self.encoder_monitor.is_stalled():
             logger.warning(f"STALL PENALTY applied")
             reward -= 2.0

        sequence_reward = self.movement_tracker.get_sequence_reward()
        if sequence_reward != 0:
            reward += sequence_reward

        if final_action == Action.FORWARD and self.lidar.min_dist < 0.35:
            proximity_penalty = -2.0 * (1.0 - self.lidar.min_dist / 0.35)
            reward += proximity_penalty

        if 'features' not in locals():
              self.lorenz.step()
              free_angle, free_mag = self.instinct.compute_free_space_vector(lidar_16)
              features = self.brain.get_features(
                  lidar_16, us_left_dist, us_right_dist,
                  encoder_l, encoder_r,
                  self.lorenz.x_norm, self.lorenz.z_norm,
                  rear_bumper, self.lidar.min_dist,
                  last_action=self._last_action_type,
                  free_angle=free_angle, free_mag=free_mag,
                  slip_ratio=enc_stats['slip_ratio'],
                  is_stalled=enc_stats['is_stalled'],
                  traj_correction=enc_stats['correction'],
                  stall_intensity=enc_stats['stall_count'] / 10.0)

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
                done=False
            )

        self.concept_graph.update(final_action, reward, self.cycle_count)
        if self.cycle_count % self.config.CONCEPT_PRUNING_INTERVAL == 0 and self.cycle_count > 0:
            self.concept_graph.prune_and_merge(self.cycle_count)

        front_dist_est = self.fusion.front_dist
        aggression_factor = self.lorenz.z_norm
        fwd_velocity = self.velocity_mapper.compute_base_velocity(front_dist_est, aggression_factor)
        base_velocity = self.velocity_mapper.compute_base_velocity(self.lidar.min_dist, aggression_factor)

        if final_action == Action.FORWARD:
            target_l, target_r = fwd_velocity, fwd_velocity
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

        directional_bias = self.lorenz.x_norm
        if final_action in (Action.TURN_LEFT, Action.TURN_RIGHT, Action.SPIN_LEFT, Action.SPIN_RIGHT):
            bias_strength = self.config.LORENZ_BIAS_SCALE
            target_l += directional_bias * bias_strength
            target_r -= directional_bias * bias_strength

        target_l, target_r = self.velocity_mapper.apply_ramp_limit(target_l, target_r)

        if final_action == Action.REVERSE:
            symmetric = (target_l + target_r) / 2.0
            target_l, target_r = symmetric, symmetric

        self.brain.last_features = features
        self.brain.last_action = final_action

        pwm_l, pwm_r = self.motors.update_pid(target_l, target_r, encoder_l, encoder_r, dt)

        # SLIP CORRECTION
        if self.encoder_monitor.is_slipping() and final_action == Action.FORWARD:
             pwm_l *= 0.8
             pwm_r *= 0.8
             logger.debug(f"Slip corrected PWM")

        # TRAJECTORY CORRECTION
        if final_action == Action.FORWARD:
             correction = enc_stats['correction'] * 30.0
             pwm_l -= correction
             pwm_r += correction

             # Also old corrections?
             # The old code had:
             # diff = encoder_l - encoder_r
             # correction = diff * self.config.FORWARD_ENCODER_CORRECTION * 50.0
             # EncoderMonitor.get_trajectory_correction uses history of diffs.
             # I should use the new one.

             micro_noise = directional_bias * self.config.FORWARD_LORENZ_PWM
             pwm_l += micro_noise
             pwm_r -= micro_noise

        # Set expected for NEXT loop
        self.encoder_monitor.set_expected(pwm_l, pwm_r)

        if final_action == Action.REVERSE:
            symmetric_pwm = -abs((pwm_l + pwm_r) / 2.0)
            pwm_l, pwm_r = symmetric_pwm, symmetric_pwm
            self.motors.sync_memory(pwm_l, pwm_r)

        if final_action == Action.FORWARD:
             self.motors.sync_memory(pwm_l, pwm_r)

        self.anti_stagnation.update(self.motors.x, self.motors.y, (abs(pwm_l)+abs(pwm_r))/2.0, final_action)
        if final_action in (Action.TURN_LEFT, Action.TURN_RIGHT, Action.SPIN_LEFT, Action.SPIN_RIGHT):
             pwm_l, pwm_r = self.anti_stagnation.inject_chaos(
                 self.lorenz.x_norm, self.lorenz.z_norm, pwm_l, pwm_r
             )

        pwm_l = np.clip(pwm_l, -self.config.PWM_MAX, self.config.PWM_MAX)
        pwm_r = np.clip(pwm_r, -self.config.PWM_MAX, self.config.PWM_MAX)
        # 22. Pelna diagnostyka co 50 cykli - WERSJA PREYZYJNA!
        if self.cycle_count % 50 == 0:
            # Bezpieczne pobieranie wartości
            q_vals = self.brain.q_approx.predict_q(features) # Uzywamy q_approx bezposrednio
            gate_weights = np.zeros(2) # Brak gate w DualLinear

            # Formatowanie Q - zawsze pokazuj min i max
            q_min = np.min(q_vals)
            q_max = np.max(q_vals)
            q_norm = np.linalg.norm(q_vals)

            # Jedna, czysta linia logu - DOKŁADNIE TAK JAK W TWOIM PRZYKŁADZIE!
            logger.info(
                f"[DIAG] c={self.cycle_count} "
                f"PWM=({pwm_l:+.0f},{pwm_r:+.0f}) "
                f"USL={us_left_dist:.2f} USR={us_right_dist:.2f} "
                f"Lmin={self.lidar.min_dist:.2f} "
                f"act={final_action.name} src={source} "
                f"Lx={directional_bias:+.2f} Lz={aggression_factor:.2f} "
                f"front_clr={front_clearance:.2f} "
                f"free_ang={math.degrees(free_angle):+.0f}deg "
                f"mag={free_mag:.2f} "
                f"stag={'STAGNANT' if self.anti_stagnation.is_stagnant else 'ok'} "
                f"Q=[{q_min:+.2f},{q_max:+.2f}] "
                f"Qnrm={q_norm:.1f} "
                f"eps={self.brain.epsilon:.3f} "
                f"lr={self.brain.lr:.5f} "
                f"buf={len(self.brain.replay_buffer)} "
                f"CF=0" # Brak CF w tej wersji
            )

        return pwm_l, pwm_r
        # POZIOM 1: Damper handled in loop
        # POZIOM 2: Blokada
        if self.stuck_detector.update(self.fusion.front_dist, enc_l, enc_r):
            logger.warning("STUCK DETECTED! Uwalniam robota...")
            if self.fusion.rear_dist > 0.3:
                return Action.REVERSE, "STUCK_REVERSE"
            else:
                if self.fusion.left_dist >= self.fusion.right_dist:
                    return Action.SPIN_LEFT, "STUCK_SPIN_LEFT"
                else:
                    return Action.SPIN_RIGHT, "STUCK_SPIN_RIGHT"

        # POZIOM 3: Bezpieczenstwo
        if self.fusion.front_dist < self.config.LIDAR_HARD_SAFETY_MIN:
             if self.fusion.rear_dist > 0.3:
                  return Action.REVERSE, "SAFETY_REVERSE"
             else:
                  if self.fusion.left_dist >= self.fusion.right_dist:
                       return Action.SPIN_LEFT, "SAFETY_SPIN"
                  else:
                       return Action.SPIN_RIGHT, "SAFETY_SPIN"
        return None, "NORMAL"


    def get_stats(self) -> Dict:
        return {
            'cycle_count':       self.cycle_count,
            'epsilon':           self.brain.epsilon,
            'lr':                self.brain.lr,
            'replay_size':       len(self.brain.replay_buffer),
            'concepts_count':    len(self.concept_graph.concepts),
            'stagnant':          self.anti_stagnation.is_stagnant,
        }

SwarmCoreV54 = SwarmCoreV55
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SWARM CORE v5.5 -- DUAL LINEAR BRAIN -- TEST")
    print("=" * 70 + "\n")

    cfg = SwarmConfig()
    core = SwarmCoreV55()

    print("[INTEG] SwarmCoreV55 DualLinear -- 20 krokow ...")
    for i in range(20):
        pwm_l, pwm_r = core.loop(
            lidar_points=[(a, max(0.1, 2.0 - i*0.1)) for a in range(0, 360, 22)],
            encoder_l=0.3, encoder_r=0.3, motor_current=1.2,
            us_left_dist=max(0.1, 2.0 - i*0.1), us_right_dist=max(0.1, 2.0 - i*0.1), dt=0.1
        )
        print(f"#{i+1:2d}: PWM=({pwm_l:+6.1f},{pwm_r:+6.1f}) | {core.brain.last_action}")

    core.save_state()
    print("\n[OK] Test PASSED!")
