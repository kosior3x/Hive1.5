
import numpy as np
import random

class SemanticEngineMock:
    def __init__(self, dim=16): # Keeping 16D to match current agents, scalable to 320D
        self.dim = dim
        self.categories = {
            'animals': self._random_vec(),
            'vehicles': self._random_vec(),
            'emotions': self._random_vec(),
            'nature': self._random_vec(),
            'conversation': self._random_vec()
        }
        self.vocab = {
            'kot': 'animals', 'pies': 'animals', 'koń': 'animals', 'krowa': 'animals',
            'samochód': 'vehicles', 'rower': 'vehicles', 'autobus': 'vehicles', 'pociąg': 'vehicles',
            'radość': 'emotions', 'smutek': 'emotions', 'gniew': 'emotions',
            'drzewo': 'nature', 'kwiat': 'nature', 'rzeka': 'nature', 'zielony': 'nature',
            'witaj': 'conversation', 'cześć': 'conversation', 'dzień': 'conversation',
            'kamil': 'conversation', 'tato': 'conversation', 'imię': 'conversation', 'decyzja': 'conversation',
            'jestem': 'conversation', 'to': 'conversation',
            # Common Pronouns & Verbs
            'ja': 'conversation', 'ty': 'conversation', 'my': 'conversation', 'wy': 'conversation',
            'on': 'conversation', 'ona': 'conversation', 'ono': 'conversation',
            'jesteś': 'conversation', 'jest': 'conversation', 'są': 'conversation', 'być': 'conversation',
            'mam': 'conversation', 'masz': 'conversation', 'ma': 'conversation',
            'wiem': 'conversation', 'wiesz': 'conversation', 'rozumiem': 'conversation', 'chcę': 'conversation',
            # Common Particles/Adverbs
            'tak': 'conversation', 'nie': 'conversation', 'może': 'conversation', 'chyba': 'conversation',
            'bardzo': 'conversation', 'trochę': 'conversation', 'dużo': 'conversation', 'mało': 'conversation',
            'tutaj': 'conversation', 'tam': 'conversation', 'teraz': 'conversation', 'kiedyś': 'conversation',
            'się': 'conversation', 'o': 'conversation', 'w': 'conversation', 'z': 'conversation', 'na': 'conversation',
            'że': 'conversation', 'ale': 'conversation', 'i': 'conversation', 'a': 'conversation',
            # Positive Emotions/Feedback
            'lubię': 'emotions', 'kocham': 'emotions', 'uwielbiam': 'emotions', 'szanuję': 'emotions',
            'wspaniale': 'emotions', 'miło': 'emotions', 'super': 'emotions', 'ekstra': 'emotions',
            'inspiracja': 'emotions', 'ciekawe': 'emotions', 'dzięki': 'emotions', 'proszę': 'emotions',
            # Capabilities & Questions
            'potrafisz': 'conversation', 'umiejętności': 'conversation', 'umiesz': 'conversation',
            'możesz': 'conversation', 'możliwości': 'conversation', 'funkcjonować': 'conversation',
            'funkcjonuje': 'conversation', 'jak': 'conversation', 'co': 'conversation', 'dlaczego': 'conversation',
            'kiedy': 'conversation', 'gdzie': 'conversation', 'twój': 'conversation', 'twoja': 'conversation',
            'twoje': 'conversation', 'mój': 'conversation', 'moja': 'conversation', 'moje': 'conversation',
            'dzisiaj': 'conversation', 'dziś': 'conversation', 'jutro': 'conversation', 'wczoraj': 'conversation',
            'chaos': 'conversation', 'system': 'conversation', 'pamięć': 'conversation', 'rozmawiać': 'conversation'
        }

    def _random_vec(self):
        v = np.random.randn(self.dim)
        return v / np.linalg.norm(v)

    def encode(self, text):
        # Dynamic Weighted Bag-of-Words (Simulating Attention)
        words = text.lower().replace('.', '').replace(',', '').replace('!', '').replace('?', '').split()

        weighted_sum = np.zeros(self.dim)
        total_weight = 0.0

        for w in words:
            cat = self.vocab.get(w, None)
            if cat:
                # High attention to known concepts
                weight = 3.0
                base = self.categories[cat]
                noise = np.random.randn(self.dim) * 0.1
                vec = base + noise
                normalized_vec = vec / np.linalg.norm(vec)

                weighted_sum += normalized_vec * weight
                total_weight += weight
            else:
                # Low attention to background noise/unknown words
                weight = 0.5
                np.random.seed(sum([ord(c) for c in w]))
                v = np.random.randn(self.dim)
                normalized_vec = v / np.linalg.norm(v)

                weighted_sum += normalized_vec * weight
                total_weight += weight

        if total_weight == 0:
            return np.zeros(self.dim)

        # Weighted Average
        avg_vec = weighted_sum / total_weight
        return avg_vec / (np.linalg.norm(avg_vec) + 1e-9)

    def decode_category(self, vec):
        best_cat = None
        max_sim = -1.0

        # Normalize input
        n_vec = vec / (np.linalg.norm(vec) + 1e-9)

        for cat, center in self.categories.items():
            sim = np.dot(n_vec, center)
            # Normalize to [0, 1] range
            sim = (sim + 1) / 2.0
            if sim > max_sim:
                max_sim = sim
                best_cat = cat
        return best_cat, max_sim
