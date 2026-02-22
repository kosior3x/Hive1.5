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
        self.concept_graph = ConceptGraph(self.config)  # ★ WŁAŚCIWY!

        # Moduly
        self.lorenz = LorenzAttractor(self.config)
        self.brain.lorenz = self.lorenz
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

        # Rear bumper state (Legacy field, kept for compatibility if needed, but BumperSystem should handle it)
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
