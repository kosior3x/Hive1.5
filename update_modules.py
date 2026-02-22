class StuckDetector:
    """
    Wykrywa czy robot jest zablokowany.
    Koła się kręcą (enkodery > 0), ale odległość z przodu nie maleje.
    """
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.front_dist_history = deque(maxlen=10)
        self.stuck_count = 0
        self.last_stuck_cycle = 0
        self.stuck_cooldown = 50

    def update(self, front_dist: float, enc_l: float, enc_r: float, current_cycle: int = 0) -> bool:
        if current_cycle - self.last_stuck_cycle < self.stuck_cooldown:
            return False

        self.front_dist_history.append(front_dist)

        wheels_moving = abs(enc_l) > 0.02 or abs(enc_r) > 0.02

        if not wheels_moving:
            self.stuck_count = 0
            return False

        if len(self.front_dist_history) == 10:
            dist_std = np.std(self.front_dist_history)
            if dist_std < 0.01:
                self.stuck_count += 1
                if self.stuck_count > 8:
                    self.last_stuck_cycle = current_cycle
                    return True
            else:
                self.stuck_count = max(0, self.stuck_count - 1)

        return False

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
            # Handle variable angle input (real hardware)
            sector = int((angle % 360) / 22.5) % 16
            sector_dists[sector].append(dist)

        self.min_dist = self.config.LIDAR_MAX_RANGE
        for i, dists in enumerate(sector_dists):
            if dists:
                min_d = min(dists) # Safety: closest object
                self.min_dist = min(self.min_dist, min_d)
                self.sectors_16[i] = 1.0 - min(min_d / self.config.LIDAR_MAX_RANGE, 1.0)
            else:
                self.sectors_16[i] = 0.0

        return self.sectors_16

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
        # US dają odległość na wprost z offsetem kątowym
        us_min = min(us_left, us_right)

        # LIDAR daje dokładny pomiar na wprost (sektory 15 i 0)
        lidar_front_central = min(lidar_16[15], lidar_16[0])
        # Convert from normalized to meters: 1.0 -> 0m, 0.0 -> max_range
        lidar_front_dist = (1.0 - lidar_front_central) * self.config.LIDAR_MAX_RANGE

        # Fuzja: bierzemy MNIEJSZĄ wartość (bezpieczeństwo!)
        self.front_dist = min(us_min, lidar_front_dist)

        # Rear (sektory 7, 8)
        lidar_rear = min(lidar_16[7], lidar_16[8])
        self.rear_dist = (1.0 - lidar_rear) * self.config.LIDAR_MAX_RANGE

        # Left (sektory 11, 12)
        lidar_left = min(lidar_16[11], lidar_16[12])
        self.left_dist = (1.0 - lidar_left) * self.config.LIDAR_MAX_RANGE

        # Right (sektory 3, 4)
        lidar_right = min(lidar_16[3], lidar_16[4])
        self.right_dist = (1.0 - lidar_right) * self.config.LIDAR_MAX_RANGE
