    def loop(self, lidar_points: List[Tuple[float, float]],
             encoder_l: float, encoder_r: float,
             motor_current: float,
             us_left_dist: float = 3.0, us_right_dist: float = 3.0,
             rear_bumper: int = 0,
             dt: float = 0.033) -> Tuple[float, float]:

        self.cycle_count += 1

        # === INICJALIZACJA ZMIENNYCH (ZAPOBIEGA BŁĘDOM) ===
        front_clearance = 1.0
        free_angle = 0.0
        free_mag = 0.0
        directional_bias = 0.0
        aggression_factor = 0.0
        features = None
        # =================================================

        # 0. Update Encoder Monitor (Observe effect of previous action)
