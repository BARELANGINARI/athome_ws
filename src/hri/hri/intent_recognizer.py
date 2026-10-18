"""
Intent Recognition Module (NLU).
Menggunakan fuzzy semantic sequence matching untuk mendeteksi intent dari transkripsi teks.
"""

from difflib import SequenceMatcher


class IntentRecognizer:
    """Modul khusus analisis teks untuk mencocokkan intensi perintah operator."""

    def __init__(self, threshold: float = 0.68):
        self.threshold = threshold
        self.intent_corpus = {
            "WAKEWORD": [
                "hello robot",
                "halo robot",
                "hey robot",
                "hello barelang",
                "hi robot",
                "robot",
            ],
            "FOLLOW_ME": [
                "follow me",
                "please follow me",
                "come with me",
                "lets go follow me",
                "start following me",
                "guide me follow me",
                "following me",
            ],
            "ARRIVED": [
                "arrived at the location",
                "we have arrived",
                "this is the location",
                "reached the car",
                "arrived at the car",
                "stop here we arrived",
                "i have arrived",
                "we are here",
                "arrived",
            ],
        }

    def match_intent(self, transcript: str, target_intent: str) -> bool:
        """Memeriksa apakah teks transkripsi cocok dengan target intent."""
        if not transcript or not transcript.strip():
            return False

        text_clean = transcript.lower().strip()
        candidates = self.intent_corpus.get(target_intent, [])
        best_score = 0.0

        for phrase in candidates:
            score = SequenceMatcher(None, text_clean, phrase).ratio()
            if phrase in text_clean:
                score = max(score, 0.95)
            best_score = max(best_score, score)

        print(f"[NLU] Input: '{transcript}' | Target: {target_intent} | Similarity: {best_score:.2f}")
        return best_score >= self.threshold

    def get_intent(self, transcript: str) -> str:
        """Mengekstrak intent terbaik dari teks, atau mengembalikan None jika tidak cocok."""
        if not transcript or not transcript.strip():
            return None

        for intent in ["WAKEWORD", "FOLLOW_ME", "ARRIVED"]:
            if self.match_intent(transcript, intent):
                return intent
        return None

