import re

file_path = 'Core/swarm_core_v5_5.py'
with open(file_path, 'r') as f:
    content = f.read()

# 1. Update NeuralHybridBrain.__init__
init_pattern = r"(class NeuralHybridBrain:.*?def __init__\(self, config: SwarmConfig, feature_extractor: FeatureExtractor\):)(.*?)(        # Standardowe komponenty)"
init_replacement = r"""\1
        self.config = config
        self.feature_extractor = feature_extractor
        self.actions_list = list(Action)
        self.n_actions = len(self.actions_list)

        # Inicjalizacja feature extractor z dummy data zeby poznac n_features
        dummy = self.feature_extractor.extract(
            np.zeros(16), 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0, 3.0, None, 0.0, 0.0
        )
        self.n_features = len(dummy) # Powinno byc 82

        # Krok 2: Uzywamy DualLinearApproximator zamiast NeuralBrain
        self.q_approx = DualLinearApproximator(
            n_features=self.n_features,
            n_actions=self.n_actions,
            learning_rate=config.LEARNING_RATE,
            avoidance_penalty=config.AVOIDANCE_PENALTY,
            weight_clip=10.0
        )

        self.lr = config.NN_LEARNING_RATE

\3"""

content = re.sub(init_pattern, init_replacement, content, flags=re.DOTALL)

# 2. Update NeuralHybridBrain.decide
decide_pattern = r"def decide\(self, features: np.ndarray, instinct_bias: Dict\[Action, float\],\s+concept_suggestion: Optional\[Action\]\) -> Tuple\[Action, str, np.ndarray\]:.*?(?=    def is_bad_state)"
decide_replacement = r"""def decide(self, features: np.ndarray, instinct_bias: Dict[Action, float],
               concept_suggestion: Optional[Action]) -> Tuple[Action, str, np.ndarray]:

        if random.random() < self.epsilon:
            return random.choice(self.actions_list), "EXPLORE", np.zeros(2)

        combined_q = self.q_approx.predict(features)   # juz z uwzglednieniem A
        scores = combined_q.copy()

        for i, action in enumerate(self.actions_list):
            scores[i] += instinct_bias.get(action, 0.0) * 4.0
            if action == concept_suggestion:
                scores[i] += 1.5

        best_idx = int(np.argmax(scores))
        return self.actions_list[best_idx], "Q_APPROX+INST+CONCEPT", np.zeros(2)

"""
content = re.sub(decide_pattern, decide_replacement, content, flags=re.DOTALL)


# 3. Update NeuralHybridBrain.is_bad_state
bad_state_pattern = r"def is_bad_state\(self, reward: float, source: str, action: Action,\s+lidar_min: float, stagnant: bool, oscillated: bool\) -> Tuple\[bool, float\]:.*?(?=    def update_q)"
bad_state_replacement = r"""def is_bad_state(self, reward: float, source: str, action: Action,
                 lidar_min: float, stagnant: bool, oscillated: bool) -> Tuple[bool, float]:
        # Kolizja / twardy odruch (ale nie gdy sam REVERSE)
        if source in ("LIDAR_HARD_SAFETY", "HARD_REFLEX") and action != Action.REVERSE:
            return True, 2.0   # bardzo zle

        # Stagnacja – utkniecie (ale nie podczas celowego skretu)
        if stagnant and action in (Action.FORWARD, Action.REVERSE):
            return True, 1.5

        # Oscylacja – wykryta przez anti‑oscillation
        if oscillated:
            return True, 1.2

        # Bardzo niska nagroda (np. -2.0 z VirtualDamper)
        if reward < -1.5:
            return True, 1.0

        # Jazda do przodu zbyt blisko sciany (ale jeszcze nie kolizja)
        if action == Action.FORWARD and lidar_min < 0.2:
            return True, 0.8

        return False, 0.0

"""
content = re.sub(bad_state_pattern, bad_state_replacement, content, flags=re.DOTALL)


# 4. Update NeuralHybridBrain.update_q (and remove complex logic)
# This is tricky because update_q is large. Let's find end of class or next method.
# In the file, update_q is followed by .
update_q_pattern = r"def update_q\(self, old_features: np.ndarray, action: Action,.*?def _train_on_batch"
update_q_replacement = r"""def update_q(self, old_features: np.ndarray, action: Action,
                 reward: float, new_features: np.ndarray,
                 source: str, lidar_min: float,
                 stagnant: bool, oscillated: bool,
                 done: bool = False, lr_scale=1.0):

        action_idx = self.actions_list.index(action)

        # Czy to zle doswiadczenie?
        is_bad, bad_target = self.is_bad_state(reward, source, action,
                                                lidar_min, stagnant, oscillated)

        if is_bad:
            # Aktualizuj macierz unikania
            self.q_approx.update_a(old_features, action_idx, bad_target)
        else:
            # Normalna aktualizacja Q (TD)
            if done:
                target_q = reward
            else:
                next_q = self.q_approx.predict_q(new_features)
                max_next_q = float(np.max(next_q))
                target_q = reward + self.config.DISCOUNT_FACTOR * max_next_q
            self.q_approx.update_q(old_features, action_idx, target_q)

        # Decay epsilon i LR
        self.epsilon = max(self.config.EPSILON_MIN,
                           self.epsilon * self.config.EPSILON_DECAY)
        self.lr = max(self.config.LR_MIN,
                      self.lr * self.config.LR_DECAY)
        self.q_approx.set_learning_rate(self.lr)

    def _train_on_batch""" # Keeping the start of next method to match regex end group

content = re.sub(update_q_pattern, update_q_replacement, content, flags=re.DOTALL)


# 5. Remove _train_on_batch and complex L2/NN methods that are no longer called?
# For now, I leave them as dead code or just the ones I replaced.
# But I need to update save/load in SwarmCoreV55, which is further down.

# 6. Update SwarmCoreV55.save_state
save_state_pattern = r"def save_state\(self\):.*?(?=        # 2. Przygotuj dane)"
save_state_replacement = r"""def save_state(self):
"""
content = re.sub(save_state_pattern, save_state_replacement, content, flags=re.DOTALL)

# Insert q_weights/a_weights in data dict
data_pattern = r"('lorenz_state': self.lorenz.get_state\(\),)"
data_replacement = r"""\1
            'q_weights': self.brain.q_approx.q_weights.tolist(),
            'a_weights': self.brain.q_approx.a_weights.tolist(),"""
content = re.sub(data_pattern, data_replacement, content)

# 7. Update SwarmCoreV55._load_state
load_state_pattern = r"(self.brain.normalizer.set_state\(norm_state\)\s+logger.info\(f\"Loaded normalizer: n={norm_state\['n'\]} samples\"\))"
load_state_replacement = r"""\1

            if 'q_weights' in saved and saved['q_weights'] is not None:
                self.brain.q_approx.q_weights = np.array(saved['q_weights'])
            if 'a_weights' in saved and saved['a_weights'] is not None:
                self.brain.q_approx.a_weights = np.array(saved['a_weights'])"""
content = re.sub(load_state_pattern, load_state_replacement, content)

# 8. Update NeuralHybridBrain.__init__ again to remove atexit.register(self.save) if it was there?
# It was in the regex \3 group which I kept.
# But self.save method in NeuralHybridBrain (which saves "hierarchical_swarm_mlp_v6.npz") is now broken because self.nn is gone.
# I should probably remove the atexit registration or update self.save.
# The prompt implies using SwarmCoreV55.state_manager for saving everything in brain_v5_5.pkl.
# So I should probably remove the separate brain saving mechanism from NeuralHybridBrain.

# Let's remove atexit from NeuralHybridBrain init
atexit_pattern = r"        # Rejestracja auto-zapisu\s+import atexit\s+atexit.register\(self.save\)"
content = re.sub(atexit_pattern, "", content)

# And clean up  etc from init if they cause errors (they are removed in regex 1).

with open(file_path, 'w') as f:
    f.write(content)
