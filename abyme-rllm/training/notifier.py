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
            if url is None:
                raise ValueError("Discord webhook URL must be provided.")

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
        
        try:
            requests.post(self.url, json={"content": formatted_message})
            self._last_send_time = current_time
        except Exception as e:
            # We print to stderr as a fallback if Discord fails, 
            # but we don't crash the program.
            sys.__stderr__.write(f"[Notifier Error] Could not send to Discord: {e}\n")

    def _format_message(self, message: str) -> str:
        if self.identity:
            return f"**{self.identity}:** {message}"
        return message

mailman = DiscordNotifier()  # Global instance for easy use throughout the codebase

class EmptyNotifier(Notifier):
    """
    A null object notifier that discards all messages.
    Useful for testing or silent operation.
    """
    def send(self, message: str, important: bool = False) -> None:
        pass

class ConsoleNotifier(Notifier):
    """
    A notifier that prints messages to the console (stdout).
    """
    def send(self, message: str, important: bool = False) -> None:
        if important:
            # Important messages get their own line
            print(message, file=sys.stdout)
        else:
            # Non-important messages (like progress bars) overwrite the current line.
            # \x1b[K is an ANSI escape code to Clear the Line from cursor to end.
            # This prevents "ghosting" where parts of a previous long message 
            # remain visible after a shorter message is printed.
            print(f"\r{message}\x1b[K", end="", file=sys.stdout, flush=True)

class CompositeNotifier(Notifier):
    """
    A notifier that broadcasts messages to a list of other notifiers.
    Example: Print to console AND send to Discord.
    """
    def __init__(self, notifiers: List[Notifier]):
        self.notifiers = notifiers

    def send(self, message: str, important: bool = False) -> None:
        for notifier in self.notifiers:
            try:
                notifier.send(message, important)
            except Exception as e:
                # Prevent one failing notifier (e.g. Discord timeout) 
                # from breaking the loop for others.
                sys.stderr.write(f"Notifier failed: {e}\n")

class StderrToNotifier:
    """
    A file-like object that captures stderr output (including tqdm progress bars)
    and forwards clean text to a Notifier instance.
    """
    def __init__(self, notifier: Notifier):
        self.notifier = notifier
        self.buffer = ""
        # Regex to remove ANSI color codes (e.g., \x1b[32m) which break Discord formatting
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def write(self, text):
        # 1. Pass through to actual console so user still sees it locally
        sys.__stderr__.write(text)
        
        # 2. Accumulate in buffer
        self.buffer += text
        
        # 3. Process lines (tqdm uses \r for updates)
        if '\r' in self.buffer or '\n' in self.buffer:
            # Clean ANSI codes
            clean_msg = self.ansi_escape.sub('', self.buffer).strip()
            self.buffer = ""
            
            if clean_msg:
                # Send with important=False to let the Notifier throttle updates
                self.notifier.send(clean_msg, important=False)

    def flush(self):
        sys.__stderr__.flush()