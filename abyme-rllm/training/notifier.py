"""
Notifier module for handling external notifications and capturing stream output.
"""
import os
import time
import sys
import re
import dotenv
import requests
from abc import ABC, abstractmethod
from typing import Optional, List

dotenv.load_dotenv()

class Notifier(ABC):
    """Abstract base class for all notifier implementations."""
    @abstractmethod
    def send(self, message: str, important: bool = False) -> None:
        pass

class DiscordNotifier(Notifier):
    """
    Discord webhook notifier with rate limiting.
    """
    def __init__(
        self,
        url: Optional[str] = None,
        identity: Optional[str] = None,
        frequency: int = 60
    ):
        if url is None:
            url = os.getenv("DISCORD_URL")

        self.url = url
        self.identity = identity
        self.frequency = frequency
        self._last_send_time: float = 0.0

    def send(self, message: str, important: bool = True) -> None:
        current_time = time.time()

        # Rate limiting for non-important messages
        if not important:
            if current_time - self._last_send_time < self.frequency:
                return

        formatted_message = self._format_message(message)
        print(formatted_message)
        self._last_send_time = current_time

        if self.url is None:
            return

        try:
            requests.post(self.url, json={"content": formatted_message})
        except Exception as e:
            # We print to stderr as a fallback if Discord fails, 
            # but we don't crash the program.
            sys.__stderr__.write(f"[Notifier Error] Could not send to Discord: {e}\n")

    def _format_message(self, message: str) -> str:
        if self.identity:
            return f"**{self.identity}:** {message}"
        return message

mailman = DiscordNotifier()  # Global instance for easy use throughout the codebase
