    def _decide_from_fusion(self, enc_l, enc_r) -> Tuple[Optional[Action], str]:
        """
        Podejmuje decyzje na podstawie fuzji sensorow i hierarchii.
        """
        # POZIOM 1: Damper (obslugiwany w loop, tu mozna dodac check statusu)
        if self.bumper_system.escape_sequence > 0:
             return self.bumper_system.check_collision(0), "BUMPER_ESCAPE"

        # POZIOM 2: Blokada (kola kreca, brak ruchu)
        if self.stuck_detector.update(self.fusion.front_dist, enc_l, enc_r):
            logger.warning("STUCK DETECTED! Uwalniam robota...")
            if self.fusion.rear_dist > 0.3:
                return Action.REVERSE, "STUCK_REVERSE"
            else:
                # Tyl zablokowany - obrot
                if self.fusion.left_dist >= self.fusion.right_dist:
                    return Action.SPIN_LEFT, "STUCK_SPIN_LEFT"
                else:
                    return Action.SPIN_RIGHT, "STUCK_SPIN_RIGHT"

        # POZIOM 3: Bezpieczenstwo (US/LIDAR < 15cm)
        # Uzywamy fusion.front_dist
        if self.fusion.front_dist < self.config.LIDAR_HARD_SAFETY_MIN:
             # Sprawdz tyl
             if self.fusion.rear_dist > 0.3:
                  return Action.REVERSE, "SAFETY_REVERSE"
             else:
                  # Spin w strone wolniejsza
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
        """Glowna petla decyzyjna (v5.5 - dual US + rear bumper)"""

        self.cycle_count += 1

        # 1. Process sensors & Fusion
        lidar_16 = self.lidar.process(lidar_points)
        self.fusion.update(us_left_dist, us_right_dist, lidar_16)

        avg_speed = (encoder_l + encoder_r) / 2.0
        dyn_us, dyn_lidar = self._compute_dynamic_safety(avg_speed)

        # 2. Hierarchia Decyzyjna

        # POZIOM 1: Damper
        bumper_action = self.bumper_system.check_collision(rear_bumper)
        if bumper_action:
             final_action = bumper_action
             source = "BUMPER_PRIORITY"
             self.stabilizer.force_unlock()
             gate_weights = np.zeros(2)
        else:
             # POZIOM 2 & 3: Blokada i Safety (via _decide_from_fusion)
             override_action, override_source = self._decide_from_fusion(encoder_l, encoder_r)

             if override_action:
                  final_action = override_action
                  source = override_source
                  self.stabilizer.force_unlock()
                  gate_weights = np.zeros(2)
             else:
                  # POZIOM 4: Normalna jazda (Brain)

                  # 3. Lorenz step
                  self.lorenz.step()
                  aggression_factor = self.lorenz.z_norm
                  directional_bias = self.lorenz.x_norm

                  # 4. Free space instinct
                  free_angle, free_mag = self.instinct.compute_free_space_vector(lidar_16)

                  front_occ     = float(np.mean([lidar_16[14], lidar_16[15],
                                                 lidar_16[0],  lidar_16[1]]))
                  front_clearance = 1.0 - front_occ

                  instinct_bias = self.instinct.get_bias_for_action(
                      free_angle,
                      magnitude=free_mag,
                      front_clearance=front_clearance,
                      us_left=us_left_dist,
                      us_right=us_right_dist
                  )
                  instinct_bias = self.instinct.apply_us_bias(instinct_bias, us_left_dist, us_right_dist)

                  # 5. Feature extraction
                  features = self.brain.get_features(
                      lidar_16, us_left_dist, us_right_dist,
                      encoder_l, encoder_r,
                      self.lorenz.x_norm, self.lorenz.z_norm,
                      rear_bumper, self.lidar.min_dist,
                      last_action=self._last_action_type,
                      free_angle=free_angle, free_mag=free_mag)

                  # 6. Concept Graph
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

                  # 7. Decision
                  # Anti-stagnation force turn check (part of Poziom 2 logic but handled by brain/stabilizer usually)
                  forced_turn = self.anti_stagnation.should_force_turn()
                  if forced_turn is not None:
                      final_action = forced_turn
                      source = "STAGNATION_FORCE"
                      self.stabilizer.force_unlock()
                      gate_weights = np.zeros(2)
                  else:
                      action_candidate, source, gate_weights = self.brain.decide(features, instinct_bias, concept_suggestion)
                      final_action = self.stabilizer.update(action_candidate)

        # ★ Anti-oscillation (Level 2.5)
        if final_action == self._last_action_type:
            self._action_repeat_count += 1
        else:
            self._action_repeat_count = 0
            self._last_action_type = final_action

        # 8. Movement Tracker & Oscillation Check
        self.motors.update_odometry(encoder_l, encoder_r, dt)
        self.movement_tracker.update(
            final_action,
            self.motors.x,
            self.motors.y,
            self.cycle_count
        )

        if self.movement_tracker.is_oscillating() and source not in ["BUMPER_PRIORITY", "STUCK_REVERSE", "STUCK_SPIN_LEFT", "STUCK_SPIN_RIGHT"]:
            logger.warning("OSCYLACJA wykryta! Wymuszam SPIN.")
            if self.fusion.left_dist >= self.fusion.right_dist:
                final_action = Action.SPIN_LEFT
            else:
                final_action = Action.SPIN_RIGHT
            source = "ANTI_OSCILLATION"
            self.stabilizer.force_unlock()

        # 9. Reward calculation
        reward = self.damper.compute_reward(encoder_l, encoder_r, motor_current, final_action)

        # Sequence reward
        sequence_reward = self.movement_tracker.get_sequence_reward()
        if sequence_reward != 0:
            reward += sequence_reward
            logger.debug(f"Sequence reward: {sequence_reward:.2f}")

        # Proximity penalty
        if final_action == Action.FORWARD and self.lidar.min_dist < 0.35:
            proximity_penalty = -2.0 * (1.0 - self.lidar.min_dist / 0.35)
            reward += proximity_penalty

        # 10. Q-Update
        # We need 'features' for update. If we took override action, we might not have features calculated?
        # We should calculate features anyway if we want to learn from override actions (Avoidance learning).
        # Or we skip update if override?
        # The Step 6 logic says "Avoidance learning is implemented... via backward_q updating Avoidance head".
        # So we should update even if override.

        if 'features' not in locals():
             # Calculate features if not calculated (e.g. override)
              # 3. Lorenz step (if not done)
              self.lorenz.step()
              # 4. Free space instinct (if not done)
              free_angle, free_mag = self.instinct.compute_free_space_vector(lidar_16)

              features = self.brain.get_features(
                  lidar_16, us_left_dist, us_right_dist,
                  encoder_l, encoder_r,
                  self.lorenz.x_norm, self.lorenz.z_norm,
                  rear_bumper, self.lidar.min_dist,
                  last_action=self._last_action_type,
                  free_angle=free_angle, free_mag=free_mag)

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

        # 11. Concept Graph Update
        self.concept_graph.update(final_action, reward, self.cycle_count)

        # ---- KONSOLIDACJA KONCEPTÓW ----
        if self.cycle_count % self.config.CONCEPT_PRUNING_INTERVAL == 0 and self.cycle_count > 0:
            self.concept_graph.prune_and_merge(self.cycle_count)

        # 12. Velocity mapping
        # Use fusion front_dist instead of instinct heuristic for base velocity?
        # Instinct uses 'front_clearance' from LIDAR. Fusion uses min(US, LIDAR).
        # Better use Fusion for safety.

        front_dist_est = self.fusion.front_dist
        # agg factor from lorenz
        aggression_factor = self.lorenz.z_norm

        fwd_velocity   = self.velocity_mapper.compute_base_velocity(front_dist_est, aggression_factor)
        base_velocity  = self.velocity_mapper.compute_base_velocity(self.lidar.min_dist, aggression_factor)

        # 13. Action -> Target velocities
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

        # 14. Lorenz bias (only for turns/spins)
        directional_bias = self.lorenz.x_norm
        if final_action in (Action.TURN_LEFT, Action.TURN_RIGHT,
                            Action.SPIN_LEFT, Action.SPIN_RIGHT):
            bias_strength = self.config.LORENZ_BIAS_SCALE
            target_l += directional_bias * bias_strength
            target_r -= directional_bias * bias_strength

        # 15. Ramp limiter
        target_l, target_r = self.velocity_mapper.apply_ramp_limit(target_l, target_r)

        # 16. Enforce symmetry
        if final_action == Action.REVERSE:
            symmetric = (target_l + target_r) / 2.0
            target_l, target_r = symmetric, symmetric

        if final_action == Action.ESCAPE_MANEUVER:
            mag = min(abs(target_l), abs(target_r))
            target_l, target_r = mag, -mag

        # 17. Update memory
        self.brain.last_features = features
        self.brain.last_action = final_action

        # 18. PID Control
        pwm_l, pwm_r = self.motors.update_pid(target_l, target_r, encoder_l, encoder_r, dt)

        # 19. Sync memory for reverse
        if final_action == Action.REVERSE:
            symmetric_pwm = -abs((pwm_l + pwm_r) / 2.0)
            pwm_l, pwm_r = symmetric_pwm, symmetric_pwm
            self.motors.sync_memory(pwm_l, pwm_r)

        # 20. Anti-stagnation chaos (only for turns)
        self.anti_stagnation.update(self.motors.x, self.motors.y, (abs(pwm_l)+abs(pwm_r))/2.0, final_action)
        if final_action in (Action.TURN_LEFT, Action.TURN_RIGHT,
                            Action.SPIN_LEFT, Action.SPIN_RIGHT):
             pwm_l, pwm_r = self.anti_stagnation.inject_chaos(
                 self.lorenz.x_norm, self.lorenz.z_norm, pwm_l, pwm_r
             )

        # 21. FORWARD corrections (Straight line via encoders)
        if final_action == Action.FORWARD:
             # Encoder correction
             # encoder_l/r assumed to be speed.
             diff = encoder_l - encoder_r
             # If left is faster, diff > 0. We need to slow down left or speed up right.
             # Correction factor
             correction = diff * self.config.FORWARD_ENCODER_CORRECTION * 50.0 # Gain
             pwm_l -= correction
             pwm_r += correction

             # Micro-noise from Lorenz
             micro_noise = directional_bias * self.config.FORWARD_LORENZ_PWM
             pwm_l += micro_noise
             pwm_r -= micro_noise

             self.motors.sync_memory(pwm_l, pwm_r)

        # 22. Auto-save
        # (Managed by brain atexit or step counter)

        # 23. Diagnostics
        if self.cycle_count % 100 == 0:
             q_norm = np.linalg.norm(self.brain.nn.W_q)
             if q_norm > 1000:
                  logger.warning(f"Q norm: {q_norm:.2f}")
             if q_norm > 10000:
                  self.brain.nn.W_q = np.random.randn(16, 8) * 0.01
                  self.brain.nn.b_q = np.zeros(8)
                  logger.critical("Q layer reset due to explosion")

        if self.cycle_count % 50 == 0:
             # Reconstruct log info...
             pass # Already logged inside decision block mostly
             # But here we have pwm
             logger.info(f"CYCLE {self.cycle_count} PWM=({pwm_l:.0f},{pwm_r:.0f}) Act={final_action.name} Src={source}")

        # Final clamp
        pwm_l = np.clip(pwm_l, -self.config.PWM_MAX, self.config.PWM_MAX)
        pwm_r = np.clip(pwm_r, -self.config.PWM_MAX, self.config.PWM_MAX)

        return pwm_l, pwm_r
