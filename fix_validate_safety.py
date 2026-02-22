import re

with open('Core/swarm_core_v5_5.py', 'r') as f:
    content = f.read()

# Replace validate_safety_constraints with correct version
# Also remove garbage before it if any (like "min(self.config.SAFETY_LIDAR_MAX, lidar_safety))")

# Find the start of validate_safety_constraints
start_marker = "def validate_safety_constraints(self, us_left_dist: float, us_right_dist: float,"
# Find the next method (loop)
end_marker = "def loop(self,"

start_idx = content.find(start_marker)
end_idx = content.find(end_marker, start_idx)

if start_idx != -1 and end_idx != -1:
    correct_method = r'''    def validate_safety_constraints(self, us_left_dist: float, us_right_dist: float,
                                   encoder_l: float, encoder_r: float,
                                   rear_bumper: int = 0) -> Optional[Tuple[Action, str]]:
        """Safety check - TERAZ TYLKO GDY NAPRAWDĘ BLISKO!"""
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

    '''
    # Check for garbage before start_idx (like "min(self.config...")
    # Scan backwards from start_idx for "def _compute_dynamic_safety" end
    prev_method_end = content.rfind("return us_safety, lidar_safety", 0, start_idx)

    if prev_method_end != -1:
        # Include everything up to the previous method end
        # Then clean buffer zone
        # Then correct method
        # Then rest

        # Adjust prev_method_end to include the line it is on?
        # It's inside the previous method.
        # "return us_safety, lidar_safety" is the end of _compute_dynamic_safety logic.

        # Find newline after return
        newline_idx = content.find("\n", prev_method_end)

        new_content = content[:newline_idx+1] + "\n" + correct_method + content[end_idx:]

        with open('Core/swarm_core_v5_5.py', 'w') as f:
            f.write(new_content)
        print("Fixed validate_safety_constraints and removed garbage")
    else:
        print("Could not find previous method end")
else:
    print(f"Indices not found: start={start_idx}, end={end_idx}")
