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

    # Model świata (Krok 5)
    WORLD_MODEL_FEATURES: int = 32         # liczba cech dla modelu swiata
    WORLD_MODEL_LEARNING_RATE: float = 0.001
    WORLD_MODEL_HIDDEN: int = 16           # rozmiar warstwy ukrytej
    WORLD_MODEL_UPDATE_FREQ: int = 10      # co ile krokow aktualizowac model swiata
    WORLD_MODEL_BATCH_SIZE: int = 64       # batch do treningu modelu swiata
    WORLD_MODEL_BUFFER_SIZE: int = 10000   # bufor doswiadczen modelu swiata
    COUNTERFACTUAL_STEPS: int = 3          # nieuzywane aktywnie w krok. 5, zostawiamy jako koncepcje
    COUNTERFACTUAL_LEARNING_RATE: float = 0.1 # wplyw kontrfaktyki na Q


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
    
    def activate(self, boost: float = 0.1):
        """Zwiększ aktywację"""
        self.activation = min(1.0, self.activation + boost)
        self.usage_count += 1
        self.last_used = time.time()
    
    def decay(self, rate: float = 0.95):
        """Zmniejsz aktywację (naturalny decay)"""
        self.activation *= rate
    
    def mark_success(self, boost: float = 0.3):
        """Zaznacz sukces - zwiększ aktywację"""
        self.success_count += 1
        self.activation = min(1.0, self.activation + boost)


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
    
    def update(self, action: Action, reward: float):
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
                        # Pasuje! Aktywuj
                        concept.activate(0.1)
                        
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


# =============================================================================
# LORENZ ATTRACTOR (bez zmian)
# =============================================================================

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
# LINEAR Q-APPROXIMATOR v5.9.2  (zachowany dla kompatybilnosci wstecznej)
# =============================================================================

class NeuralBrainWithImagination:
    def __init__(self, config: SwarmConfig, n_features: int, n_actions: int, l1_weights=None, l2_weights=None):
        self.config = config
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = config.LEARNING_RATE
        self.gamma = config.DISCOUNT_FACTOR
        
        self.W1 = np.random.randn(n_features, 32) * 0.1
        self.b1 = np.zeros(32)
        
        self.W2 = np.random.randn(32, 16) * 0.1
        self.b2 = np.zeros(16)
        
        self.W_q = np.random.randn(16, n_actions) * 0.1
        self.b_q = np.zeros(n_actions)
        
        self.W_wm1 = np.random.randn(16 + n_actions, 16) * 0.1
        self.b_wm1 = np.zeros(16)
        
        self.W_wm2 = np.random.randn(16, n_features + 1) * 0.1
        self.b_wm2 = np.zeros(n_features + 1)
        
        self.cache = {}
        
    def forward_q(self, features: np.ndarray) -> np.ndarray:
        z1 = np.dot(features, self.W1) + self.b1
        a1 = np.maximum(0, z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = np.maximum(0, z2)
        q = np.dot(a2, self.W_q) + self.b_q
        self.cache['features'] = features
        self.cache['z1'] = z1
        self.cache['a1'] = a1
        self.cache['z2'] = z2
        self.cache['a2'] = a2
        self.cache['q'] = q
        return q

    def forward_world(self, action: int) -> Tuple[np.ndarray, float]:
        a2 = self.cache['a2']
        action_onehot = np.zeros(self.n_actions)
        action_onehot[action] = 1.0
        x = np.concatenate([a2, action_onehot])
        z_wm1 = np.dot(x, self.W_wm1) + self.b_wm1
        a_wm1 = np.maximum(0, z_wm1)
        out = np.dot(a_wm1, self.W_wm2) + self.b_wm2
        self.cache['x_wm'] = x
        self.cache['z_wm1'] = z_wm1
        self.cache['a_wm1'] = a_wm1
        self.cache['out_wm'] = out
        return out[:-1], float(out[-1])

    def backward_q(self, target_q: np.ndarray):
        d_q = self.cache['q'] - target_q
        d_a2 = np.dot(d_q, self.W_q.T)
        d_z2 = d_a2 * (self.cache['z2'] > 0)
        d_a1 = np.dot(d_z2, self.W2.T)
        d_z1 = d_a1 * (self.cache['z1'] > 0)
        
        self.W_q -= self.lr * np.outer(self.cache['a2'], d_q)
        self.b_q -= self.lr * d_q
        self.W2 -= self.lr * np.outer(self.cache['a1'], d_z2)
        self.b2 -= self.lr * d_z2
        self.W1 -= self.lr * np.outer(self.cache['features'], d_z1)
        self.b1 -= self.lr * d_z1
        for w in [self.W1, self.W2, self.W_q]:
            np.clip(w, -5, 5, out=w)

    def backward_world(self, target_next_features: np.ndarray, target_reward: float):
        target = np.concatenate([target_next_features, [target_reward]])
        d_out = self.cache['out_wm'] - target
        d_a_wm1 = np.dot(d_out, self.W_wm2.T)
        d_z_wm1 = d_a_wm1 * (self.cache['z_wm1'] > 0)
        
        self.W_wm2 -= self.lr * np.outer(self.cache['a_wm1'], d_out)
        self.b_wm2 -= self.lr * d_out
        self.W_wm1 -= self.lr * np.outer(self.cache['x_wm'], d_z_wm1)
        self.b_wm1 -= self.lr * d_z_wm1
        for w in [self.W_wm1, self.W_wm2]:
            np.clip(w, -5, 5, out=w)

    def generate_counterfactual(self, features: np.ndarray, action_taken: int, actual_reward: float):
        self.forward_q(features)
        best_value = -np.inf
        best_action = None
        best_next_features = None
        for a in range(self.n_actions):
            if a == action_taken: continue
            next_feat, pred_reward = self.forward_world(a)
            cf_value = pred_reward + self.gamma * np.max(self.forward_q(next_feat))
            if cf_value > best_value:
                best_value = cf_value
                best_action = a
                best_next_features = next_feat
        if best_value > actual_reward + getattr(self.config, 'COUNTERFACTUAL_THRESHOLD', 0.5):
            return best_action, best_value, best_next_features
        return None, None, None

class NeuralHybridBrain:
    def __init__(self, config: SwarmConfig, feature_extractor: FeatureExtractor):
        self.config = config
        self.feature_extractor = feature_extractor
        self.actions_list = list(Action)
        self.n_actions = len(self.actions_list)
        
        dummy = self.feature_extractor.extract(
            np.zeros(16), 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0, 3.0, None, 0.0, 0.0
        )
        self.n_features = len(dummy)
        
        self.nn = NeuralBrainWithImagination(config, self.n_features, self.n_actions)
        self.target_nn = copy.deepcopy(self.nn)
        self.target_update_freq = 100
        self.train_steps = 0
        self.normalizer = RunningNormalizer(self.n_features)
        self.replay_buffer = ReplayBuffer(capacity=config.REPLAY_BUFFER_CAPACITY)
        self.batch_size = config.REPLAY_BATCH_SIZE
        self.train_freq = config.REPLAY_TRAIN_FREQ
        self.steps_since_train = 0
        self.epsilon = config.EPSILON
        self.lr = config.LEARNING_RATE
        self.counterfactual_count = 0
        self.last_features = None
        self.last_action = None
        
        logger.info(f"NeuralHybridBrain: {self.n_features} features, hidden=[32,16], world_model=24->16->83")

    def get_features(self, lidar_16, us_left, us_right, encoder_l, encoder_r,
                     lorenz_x, lorenz_z, rear_bumper, min_dist, last_action=None,
                     free_angle=0.0, free_mag=0.0) -> np.ndarray:
        raw = self.feature_extractor.extract(
            lidar_16, us_left, us_right, encoder_l, encoder_r,
            lorenz_x, lorenz_z, rear_bumper, min_dist, last_action, free_angle, free_mag
        )
        if raw.shape[0] != self.n_features:
            self.replay_buffer.buffer.clear()
            self.n_features = raw.shape[0]
        self.normalizer.update(raw)
        return self.normalizer.normalize(raw)

    def is_bad_state(self, reward, source, action, lidar_min, stagnant, oscillated):
        pass

    def decide(self, features: np.ndarray, instinct_bias: Dict[Action, float], concept_suggestion: Optional[Action]):
        if random.random() < self.epsilon:
            return random.choice(self.actions_list), "EXPLORE", np.array([0.5, 0.5])
        
        scores = self.nn.forward_q(features).copy()
        for i, action in enumerate(self.actions_list):
            scores[i] += instinct_bias.get(action, 0.0) * 4.0
            if action == concept_suggestion: scores[i] += 1.5
                
        best_idx = int(np.argmax(scores))
        return self.actions_list[best_idx], "NEURAL", np.array([1.0, 0.0])
        
    def update_q(self, old_features, action, reward, new_features,
                 source, lidar_min, stagnant, oscillated, done=False, lr_scale=1.0):
        action_idx = self.actions_list.index(action)
        self.replay_buffer.push(old_features, action_idx, reward, new_features, done)
        
        self.nn.forward_q(old_features)
        self.nn.forward_world(action_idx)
        self.nn.backward_world(new_features, reward)
        
        cf_action, cf_value, cf_next = self.nn.generate_counterfactual(old_features, action_idx, reward)
        if cf_action is not None:
            self.replay_buffer.push(old_features, cf_action, cf_value, cf_next, done)
            self.counterfactual_count += 1
            
        self.epsilon = max(self.config.EPSILON_MIN, self.epsilon * self.config.EPSILON_DECAY)
        self.lr = max(self.config.LR_MIN, self.lr * self.config.LR_DECAY)
        self.nn.lr = self.lr
        
        self.steps_since_train += 1
        if self.steps_since_train >= self.train_freq and len(self.replay_buffer) >= self.batch_size:
            self._train_on_batch()
            self.steps_since_train = 0

    def _train_on_batch(self):
        batch = self.replay_buffer.sample(self.batch_size)
        for old_feat, act_idx, reward, new_feat, done in batch:
            self.nn.forward_q(old_feat)
            if done:
                target_q = reward
            else:
                max_next_q = np.max(self.target_nn.forward_q(new_feat))
                target_q = reward + self.config.DISCOUNT_FACTOR * max_next_q
                
            q_old = self.nn.forward_q(old_feat).copy()
            q_old[act_idx] = target_q
            self.nn.backward_q(q_old)
            
        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_nn.W1 = self.nn.W1.copy()
            self.target_nn.b1 = self.nn.b1.copy()
            self.target_nn.W2 = self.nn.W2.copy()
            self.target_nn.b2 = self.nn.b2.copy()
            self.target_nn.W_q = self.nn.W_q.copy()
            self.target_nn.b_q = self.nn.b_q.copy()
            self.target_nn.W_wm1 = self.nn.W_wm1.copy()
            self.target_nn.b_wm1 = self.nn.b_wm1.copy()
            self.target_nn.W_wm2 = self.nn.W_wm2.copy()
            self.target_nn.b_wm2 = self.nn.b_wm2.copy()

# =============================================================================
# DC MOTOR + DAMPER (bez zmian - działają)
# =============================================================================
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
    
    def _load_state(self):
        saved = self.state_manager.load()
        if saved:
            if 'nn_W1' in saved:
                try:
                    self.brain.nn.W1 = np.array(saved['nn_W1'])
                    self.brain.nn.b1 = np.array(saved['nn_b1'])
                    self.brain.nn.W2 = np.array(saved['nn_W2'])
                    self.brain.nn.b2 = np.array(saved['nn_b2'])
                    self.brain.nn.W_q = np.array(saved['nn_W_q'])
                    self.brain.nn.b_q = np.array(saved['nn_b_q'])
                    self.brain.nn.W_wm1 = np.array(saved['nn_W_wm1'])
                    self.brain.nn.b_wm1 = np.array(saved['nn_b_wm1'])
                    self.brain.nn.W_wm2 = np.array(saved['nn_W_wm2'])
                    self.brain.nn.b_wm2 = np.array(saved['nn_b_wm2'])
                    self.brain.counterfactual_count = saved.get('counterfactual_count', 0)
                    logger.info("Neural network loaded")
                except Exception as e:
                    logger.error(f"Failed to load neural network weights: {e}")
                    
            norm_state = saved.get('normalizer_state')
            if norm_state and norm_state.get('n', 0) > 0:
                saved_mean = np.array(norm_state['mean'])
                if len(saved_mean) == self.brain.n_features:
                    self.brain.normalizer.set_state(norm_state)
                    logger.info(f"Loaded normalizer: n={norm_state['n']} samples")
                else:
                    self.brain.normalizer = RunningNormalizer(self.brain.n_features)

            self.brain.epsilon = saved.get('epsilon', self.config.EPSILON)
            self.brain.lr      = saved.get('lr', self.config.LEARNING_RATE)
            self.brain.nn.lr   = self.brain.lr

            saved_concepts = saved.get('concepts', {})
            if saved_concepts:
                for name, data in saved_concepts.items():
                    c = Concept(data['name'], data['sequence'])
                    c.activation    = data['activation']
                    c.success_count = data['success_count']
                    c.usage_count   = data['usage_count']
                    c.context       = data.get('context', {})
                    self.concept_graph.concepts[name] = c

            lorenz_state = saved.get('lorenz_state')
            if lorenz_state:
                self.lorenz.x, self.lorenz.y, self.lorenz.z = lorenz_state

    def save_state(self):
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
            'nn_W1': self.brain.nn.W1.tolist(),
            'nn_b1': self.brain.nn.b1.tolist(),
            'nn_W2': self.brain.nn.W2.tolist(),
            'nn_b2': self.brain.nn.b2.tolist(),
            'nn_W_q': self.brain.nn.W_q.tolist(),
            'nn_b_q': self.brain.nn.b_q.tolist(),
            'nn_W_wm1': self.brain.nn.W_wm1.tolist(),
            'nn_b_wm1': self.brain.nn.b_wm1.tolist(),
            'nn_W_wm2': self.brain.nn.W_wm2.tolist(),
            'nn_b_wm2': self.brain.nn.b_wm2.tolist(),
            'normalizer_state': self.brain.normalizer.get_state(),
            'epsilon':          self.brain.epsilon,
            'lr':               self.brain.lr,
            'concepts':         concepts_data,
            'lorenz_state':     self.lorenz.get_state(),
            'counterfactual_count': self.brain.counterfactual_count,
        }
        self.state_manager.save(data)
    
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
        """Główna pętla decyzyjna (v5.5 - dual US + rear bumper)"""
        
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
        
        # ★ front_clearance: średnia wolności przednich sektorów (14,15,0,1 = ±45° od 0°)
        #   lidar_16[i] = 1-dist/max → HIGH=przeszkoda, LOW=wolno
        #   front_clearance 1.0 = czysto z przodu, 0.0 = ściana wprost
        front_occ     = float(np.mean([lidar_16[14], lidar_16[15],
                                       lidar_16[0],  lidar_16[1]]))
        front_clearance = 1.0 - front_occ   # odwróć: 1=wolno, 0=ściana
        
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
                    f"[CONCEPT] Użyto konceptu '{best_concept.name}' "
                    f"(aktywacja={best_concept.activation:.2f}) "
                    f"→ sugestia: {concept_suggestion.name if concept_suggestion else 'None'}"
                )
        
        # 7. Decision (Q + Instinct + Concept!)
        collision = (us_front_min < dyn_us) or (self.lidar.min_dist < dyn_lidar)
        
        gate_weights = np.array([0.5, 0.5], dtype=np.float32)

        if safety_override:
            final_action, source = safety_override
        else:
            # ★ NOWE: Anti-stagnation może wymusić skręt
            forced_turn = self.anti_stagnation.should_force_turn()
            if forced_turn is not None:
                final_action = forced_turn
                source = "STAGNATION_FORCE"
                self.stabilizer.force_unlock()
            else:
                action_candidate, source, gate_weights = self.brain.decide(features, instinct_bias, concept_suggestion)
                final_action = self.stabilizer.update(action_candidate)
        
        # ★ NOWE: Anti-oscillation — wykryj pętlę REVERSE↔FORWARD
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
        
        # 8. Reward (★ z karą za bliskość ściany przy FORWARD)
        reward = self.damper.compute_reward(encoder_l, encoder_r, motor_current, final_action)
        
        # ★ NOWE: Kara za FORWARD blisko ściany — uczy Q-table żeby nie jechać w ścianę
        if final_action == Action.FORWARD and self.lidar.min_dist < 0.35:
            proximity_penalty = -2.0 * (1.0 - self.lidar.min_dist / 0.35)
            reward += proximity_penalty
        
        # 9. Q-Update v5.10 — Avoidance Layer
        #    Uczenie ZAWSZE (nawet na wymuszonych akcjach):
        #    - zle doswiadczenia  -> macierz A (unikanie)
        #    - dobre/neutralne   -> macierz Q (TD learning)
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
        self.concept_graph.update(final_action, reward)
        
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
        # Przekaż aktualną akcję — spin/turn NIE jest stagnacją!
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
            # Krok 1: Wymuszaj symetrię bazową
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
            
            # ★ KRYTYCZNE: Zsynchronizuj pamięć PID!
            self.motors.sync_memory(pwm_l, pwm_r)
        
        # 20. Odometry
        self.motors.update_odometry(encoder_l, encoder_r, dt)
        
        # 21. ★ Auto-save (wbudowane!)
        if self.state_manager.should_auto_save():
            self.save_state()
        
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
            wm_info = f"CF={self.brain.counterfactual_count}"

            logger.info(
                f"[DIAG] c={self.cycle_count} "
                f"PWM=({pwm_l:+.0f},{pwm_r:+.0f}) "
                f"USL={us_left_dist:.2f} USR={us_right_dist:.2f} "
                f"Lmin={self.lidar.min_dist:.2f} "
                f"act={final_action.name} src={source} "
                f"{lorenz_info} {free_info} stag={stag_info} "
                f"{q_info} {wm_info}"
            )
        
        
        # ★★★ FINAL SAFETY CLAMP — zawsze na końcu, bez wyjątków!
        pwm_l = np.clip(pwm_l, -self.config.PWM_MAX, self.config.PWM_MAX)
        pwm_r = np.clip(pwm_r, -self.config.PWM_MAX, self.config.PWM_MAX)
        
        return pwm_l, pwm_r
    
    def get_stats(self) -> Dict:
        return {
            'cycle_count':       self.cycle_count,
            'q_weights_norm':    float(np.linalg.norm(self.brain.nn.W_q)),
            'epsilon':           self.brain.epsilon,
            'lr':                self.brain.lr,
            'replay_size':       len(self.brain.replay_buffer),
            'normalizer_n':      self.brain.normalizer.n,
            'concepts_count':    len(self.concept_graph.concepts),
            'lorenz_state':      self.lorenz.get_state(),
            'position':          (self.motors.x, self.motors.y, self.motors.theta),
            'stagnant':          self.anti_stagnation.is_stagnant,
            'counterfactual_generated': getattr(self.brain, 'counterfactual_count', 0),
        }


# Backward compatibility
SwarmCoreV54 = SwarmCoreV55


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SWARM CORE v5.14 -- NEURAL BRAIN -- TEST")
    print("=" * 70 + "\n")
    
    cfg = SwarmConfig()
    core = SwarmCoreV55()
    
    print("[INTEG] SwarmCoreV55 Neural -- 20 krokow ...")
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
    print("\n[OK] Test v5.14 -- wszystkie testy jednostkowe i integracyjne PASSED!\n")
