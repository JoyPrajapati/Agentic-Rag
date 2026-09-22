"""In-memory conversation buffer with sliding window."""

from collections import defaultdict
from datetime import datetime
from typing import Dict, List


class ShortTermMemory:
    """Keeps the last N conversation turns per user."""

    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.history: Dict[str, List[Dict]] = defaultdict(list)

    def add_turn(self, user_id: str, query: str, response: str) -> None:
        """Append a turn and trim to max_turns."""
        self.history[user_id].append({
            "query": query,
            "response": response,
            "timestamp": datetime.now().isoformat(),
        })
        if len(self.history[user_id]) > self.max_turns:
            self.history[user_id] = self.history[user_id][-self.max_turns:]

    def get_history(self, user_id: str) -> List[Dict]:
        """Return conversation history for a user."""
        return self.history.get(user_id, [])

    def clear(self, user_id: str) -> None:
        """Clear history for a user."""
        if user_id in self.history:
            del self.history[user_id]

    def get_last_n_turns(self, user_id: str, n: int) -> List[Dict]:
        """Return the last N turns."""
        return self.get_history(user_id)[-n:]