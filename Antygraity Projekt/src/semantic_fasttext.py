```
import os
import numpy as np

# Requirements: pip install gensim
# Download model: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pl.300.bin.gz
# (Extract the .bin file from .gz before using)

class FastTextEngine:
    def __init__(self, model_path="cc.pl.300.bin", target_dim=16):
        print(f"⏳ Loading FastText model from {model_path} via Gensim...")
        self.model = None
        self.loaded = False

        try:
            from gensim.models import KeyedVectors
            # Gensim loads Facebook's .bin format via load_word2vec_format(..., binary=True) maybe?
            # Or load_facebook_model if full FastText is installed?
            # Actually, KeyedVectors.load_word2vec_format supports text/bin vectors but not full FastText model.
            # But Gensim has 'load_facebook_vectors' in newer versions.

            # Najbardziej uniwersalna metoda dla plików .bin (FastText):
            try:
                # Try simple KeyedVectors first (works for .vec text format)
                self.model = KeyedVectors.load_word2vec_format(model_path, binary=True)
            except:
                try:
                    # Try loading as Facebook format (requires 'gensim>=4.0')
                    # This reads the .bin file directly
                    self.model = KeyedVectors.load_facebook_model(model_path).wv
                except:
                    # Final attempt: maybe user downloaded .vec/txt version?
                    self.model = KeyedVectors.load_word2vec_format(model_path, binary=False)

            self.loaded = True
            print("✅ Model loaded successfully (Gensim).")
        except ImportError:
            print("❌ 'gensim' library not found. Please install: pip install gensim")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")

        self.target_dim = target_dim
        # Projection matrix to downscale 300D -> target_dim
        # We initialize it deterministically so restarts are consistent
        rng = np.random.RandomState(42)
        self.projection = rng.randn(300, target_dim) / np.sqrt(300)

    def encode(self, word):
        if not self.loaded or not self.model:
            return np.zeros(self.target_dim)

        try:
            # Gensim access
            if word in self.model:
                vec_300 = self.model[word]
            else:
                # OOV handling (zeros or random?)
                return np.zeros(self.target_dim)

            # Project
            if self.target_dim != 300:
                vec_out = np.dot(vec_300, self.projection)
            else:
                vec_out = vec_300

            # Normalize
            norm = np.linalg.norm(vec_out)
            if norm > 0:
                vector = vec_out / norm
                return vector
            return vec_out

        except Exception as e:
            print(f"Encode error: {e}")
            return np.zeros(self.target_dim)

    def decode_nearest(self, vec, k=1):
        # Gensim doesn't support searching by 16D projected vector easily against 300D index
        # without reversing projection (impossible).
        # So we just return a placeholder for now.
        return "N/A"

    def get_dimension(self):
        return self.target_dim
```
