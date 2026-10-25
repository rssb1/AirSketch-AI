"""Game state machine, timer, and score history. Pure Python: no UI, camera, or AI imports."""
from __future__ import annotations

import enum
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import challenges as ch

ROUND_SECONDS = 60


class GameState(enum.Enum):
    MENU = "menu"
    PLAYING = "playing"
    EVALUATING = "evaluating"
    RESULTS = "results"


@dataclass
class RoundRecord:
    challenge: str
    difficulty: str
    score: float
    seconds_used: float
    method: str


def clamp_score(score: float) -> float:
    return max(0.0, min(100.0, float(score)))


def grade_for_score(score: float) -> str:
    s = clamp_score(score)
    if s >= 85:
        return "Masterpiece!"
    if s >= 65:
        return "Great job!"
    if s >= 40:
        return "Getting there"
    if s >= 15:
        return "Needs work"
    return "Keep practicing"


class Game:
    def __init__(self, duration: float = ROUND_SECONDS, clock: Callable[[], float] = time.monotonic,
                 rng=None):
        self.duration = duration
        self._clock = clock
        self._rng = rng
        self.state = GameState.MENU
        self.difficulty = "easy"
        self.challenge: Optional[ch.Challenge] = None
        self.history: List[RoundRecord] = []
        self.skipped = 0
        self.timed_out = False
        self._deadline = 0.0
        self._started = 0.0
        self._elapsed = 0.0

    # ------------------------------------------------------------ queries
    @property
    def time_left(self) -> float:
        if self.state is GameState.PLAYING:
            return max(0.0, self._deadline - self._clock())
        if self.state is GameState.MENU:
            return self.duration
        return max(0.0, self.duration - self._elapsed)

    @property
    def can_draw(self) -> bool:
        return self.state is GameState.PLAYING and self.time_left > 0

    @property
    def rounds_completed(self) -> int:
        return len(self.history)

    @property
    def best_score(self) -> float:
        return max((r.score for r in self.history), default=0.0)

    @property
    def current_score(self) -> float:
        return self.history[-1].score if self.history else 0.0

    # ------------------------------------------------------------ transitions
    def set_difficulty(self, difficulty: str) -> None:
        if difficulty not in ch.DIFFICULTIES:
            raise ValueError(f"Unknown difficulty: {difficulty!r}")
        self.difficulty = difficulty

    def start(self, difficulty: Optional[str] = None) -> ch.Challenge:
        if difficulty:
            self.set_difficulty(difficulty)
        previous = self.challenge.key if self.challenge else None
        return self._begin(ch.pick_challenge(self.difficulty, exclude=previous, rng=self._rng))

    def _begin(self, challenge: ch.Challenge) -> ch.Challenge:
        self.challenge = challenge
        self.timed_out = False
        self._elapsed = 0.0
        self._started = self._clock()
        self._deadline = self._started + self.duration
        self.state = GameState.PLAYING
        return challenge

    def tick(self) -> bool:
        """Call every frame. Returns True exactly once, when time runs out and evaluation should begin."""
        if self.state is GameState.PLAYING and self._clock() >= self._deadline:
            self._elapsed = self.duration
            self.timed_out = True
            self.state = GameState.EVALUATING
            return True
        return False

    def submit(self) -> bool:
        """Player submits early. Returns False if a submission is already in progress / not playing."""
        if self.state is not GameState.PLAYING:
            return False
        self._elapsed = min(self.duration, self._clock() - self._started)
        self.state = GameState.EVALUATING
        return True

    def record_result(self, score: float, method: str = "unknown") -> Optional[RoundRecord]:
        """Store the evaluated score. Ignored (None) unless an evaluation is pending."""
        if self.state is not GameState.EVALUATING:
            return None
        rec = RoundRecord(self.challenge.name, self.challenge.difficulty, clamp_score(score),
                          self._elapsed, method)
        self.history.append(rec)
        self.state = GameState.RESULTS
        return rec

    def abort_evaluation(self) -> None:
        """Evaluation failed completely: let the player retry the same challenge."""
        if self.state is GameState.EVALUATING:
            self.state = GameState.RESULTS

    def skip(self) -> Optional[ch.Challenge]:
        if self.state is not GameState.PLAYING:
            return None
        self.skipped += 1
        return self.start()

    def play_again(self) -> Optional[ch.Challenge]:
        """Retry the same challenge with a fresh timer."""
        if self.state is not GameState.RESULTS or self.challenge is None:
            return None
        return self._begin(self.challenge)

    def new_challenge(self) -> Optional[ch.Challenge]:
        """Different challenge, same difficulty."""
        if self.state is not GameState.RESULTS:
            return None
        return self.start()

    def to_menu(self) -> None:
        self.state = GameState.MENU
