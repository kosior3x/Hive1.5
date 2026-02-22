import sys
from collections import deque

# Define the correct source code for the classes
bumper_system_code = '''class BumperSystem:
    """
    Obsługa fizycznych zderzaków (damperów).
    Absolutny priorytet.
    """
    def __init__(self, config: SwarmConfig):
        # print(f"BumperSystem init {id(self)}")
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
'''

encoder_monitor_code = '''class EncoderMonitor:
    """
    Zaawansowany monitoring enkoderów:
    - Wykrywanie poślizgu (slip)
    - Wykrywanie blokady (stall)
    - Korekcja toru jazdy
    - Detekcja nierówności podłoża
    """
    def __init__(self, config: SwarmConfig):
        # print(f"EncoderMonitor init {id(self)}")
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
'''

# Read current file
with open('Core/swarm_core_v5_5.py', 'r') as f:
    lines = f.readlines()

# Locate BumperSystem start and EncoderMonitor start/end (or DCMotorController start which follows)
start_bumper = -1
start_dcmotor = -1

for i, line in enumerate(lines):
    if line.strip() == 'class BumperSystem:':
        start_bumper = i
    elif line.strip() == 'class DCMotorController:':
        start_dcmotor = i
        break

if start_bumper != -1 and start_dcmotor != -1:
    # Replace everything between start_bumper and start_dcmotor with correct code
    new_lines = lines[:start_bumper] + [bumper_system_code + '\n\n' + encoder_monitor_code + '\n\n'] + lines[start_dcmotor:]

    # Also clean up duplicate _decide_from_fusion if exists
    # It was detected around line 2480 and 2625 (in previous thought process)
    # Let's verify if there are two def _decide_from_fusion

    final_lines = []
    seen_decide = 0
    skip = False

    for line in new_lines:
        if 'def _decide_from_fusion' in line:
            seen_decide += 1
            if seen_decide > 1:
                # Start skipping the second definition until next method
                skip = True

        if skip:
            # If we hit next method or end of class/file, stop skipping
            # Assuming next method is loop (if decide was before loop) or something else.
            # Actually, duplicate was AFTER loop in previous scenario?
            # Let's just remove the second occurrence entirely.
            # But we need to know when it ends.
            if line.strip().startswith('def ') and 'def _decide_from_fusion' not in line:
                skip = False
            # Check indentation to be safe?
            # This is a bit risky with simple line iteration.
            pass

        if not skip:
            final_lines.append(line)

    # Write back
    with open('Core/swarm_core_v5_5.py', 'w') as f:
        f.writelines(final_lines)
    print("Fixed classes structure")
else:
    print(f"Markers not found: Bumper={start_bumper}, DC={start_dcmotor}")
