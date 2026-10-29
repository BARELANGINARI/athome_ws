"""Natural Language Understanding (NLU) intent recognizer module."""

from difflib import SequenceMatcher
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("athome_hri.nlu")

DEFAULT_INTENT_CORPUS: Dict[str, Tuple[str, ...]] = {
    "WAKEWORD": (
        "hello robot",
        "halo robot",
        "hey robot",
        "hello barelang",
        "hi robot",
        "robot",
    ),
    "FOLLOW_ME": (
        "follow me",
        "please follow me",
        "come with me",
        "lets go follow me",
        "start following me",
        "guide me follow me",
        "following me",
    ),
    "STOP": (
        "stop",
        "stop following",
        "stop the robot",
        "please stop",
        "stop now",
    ),
    "ARRIVED": (
        "arrived at the location",
        "we have arrived",
        "this is the location",
        "reached the car",
        "arrived at the car",
        "stop here we arrived",
        "i have arrived",
        "we are here",
        "arrived",
    ),
}


class IntentRecognizer:
    """Text analysis module for matching operator command intents."""

    SUBSTRING_BONUS_SCORE: float = 0.95

    def __init__(
        self,
        threshold: float = 0.68,
        intent_corpus: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.threshold = threshold
        self.intent_corpus: Dict[str, Tuple[str, ...]] = (
            {k: tuple(v) for k, v in intent_corpus.items()}
            if intent_corpus is not None
            else dict(DEFAULT_INTENT_CORPUS)
        )

    def score_intent(self, transcript: str, target_intent: str) -> float:
        """Calculate highest similarity score (0.0 - 1.0) against target intent."""
        if not transcript or not transcript.strip():
            return 0.0

        text_clean = transcript.lower().strip()
        candidates = self.intent_corpus.get(target_intent, ())
        best_score = 0.0

        for phrase in candidates:
            score = SequenceMatcher(None, text_clean, phrase).ratio()
            if phrase in text_clean:
                score = max(score, self.SUBSTRING_BONUS_SCORE)
            best_score = max(best_score, score)

        return best_score

    def match_intent(self, transcript: str, target_intent: str) -> bool:
        """Check if transcribed text matches the target intent above threshold."""
        best_score = self.score_intent(transcript, target_intent)
        logger.info(
            "Input: '%s' | Target: %s | Similarity: %.2f",
            transcript,
            target_intent,
            best_score,
        )
        return best_score >= self.threshold

    def get_intent(self, transcript: str) -> Optional[str]:
        """Extract best matching intent from text, or return None if no match."""
        if not transcript or not transcript.strip():
            return None

        for intent in ("WAKEWORD", "FOLLOW_ME", "STOP", "ARRIVED"):
            if self.match_intent(transcript, intent):
                return intent
        return None
