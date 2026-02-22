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
