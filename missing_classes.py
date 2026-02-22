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
    Monitoruje enkodery pod kątem poślizgu i blokady.
    Liczy rzeczywistą prędkość.
    """
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.expected_l = 0.0
        self.expected_r = 0.0
        self.actual_l = 0.0
        self.actual_r = 0.0

    def set_expected(self, pwm_l: float, pwm_r: float):
        """
        Przelicz PWM na oczekiwaną prędkość [m/s].
        """
        # Liniowa aproksymacja, max speed przy 100 PWM
        # Uwzględnij deadzone silnika (np. < 20 PWM nie rusza)

        def pwm_to_speed(pwm):
            pwm_abs = abs(pwm)
            if pwm_abs < 20: return 0.0
            sign = 1.0 if pwm > 0 else -1.0
            ratio = (pwm_abs - 20) / 80.0 # 0..1
            return sign * ratio * self.config.MAX_SPEED_MPS

        self.expected_l = pwm_to_speed(pwm_l)
        self.expected_r = pwm_to_speed(pwm_r)

    def update(self, enc_l: float, enc_r: float, dt: float):
        # enc_l, enc_r to delta ticków? Czy absolutna pozycja?
        # Zwykle w loop() dostajemy pozycję lub deltę.
        # W SwarmCoreV55.loop(encoder_l, encoder_r) -> to są pozycje?
        # "encoder_l: float, encoder_r: float" -> usually current speed or position.
        # W kodzie: avg_speed = (encoder_l + encoder_r) / 2.0 -> Sugeruje prędkość [m/s] lub ticki/s.
        # DCMotorController.update_pid(..., encoder_l, ...) -> PID na prędkość.
        # Więc encoder_l/r to PRĘDKOŚĆ [m/s] lub znormalizowana.

        # Zakładamy że to prędkość m/s
        self.actual_l = enc_l
        self.actual_r = enc_r

    def get_slip_ratio(self) -> float:
        # Porównaj expected vs actual
        # Jeśli expected duże, a actual małe -> poślizg/blokada
        return 0.0 # TODO: Implementacja
