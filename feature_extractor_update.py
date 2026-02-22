        # ── 82–85: NOWE CECHY Z ENKODERÓW (EncoderMonitor) ─────────────────
        # Wymaga dostępu do self.encoder_monitor z poziomu FeatureExtractor
        # Ale FeatureExtractor nie ma referencji do EncoderMonitor...
        # Musimy przekazać te wartości jako argumenty do extract() lub
        # FeatureExtractor musi mieć dostęp do config/state.

        # W obecnej architekturze FeatureExtractor jest niezależny.
        # Dodamy argumenty do extract() w następnym kroku.
        # Tutaj tylko placeholder w komentarzu, właściwa zmiana w metodzie extract.
