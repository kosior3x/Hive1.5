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
