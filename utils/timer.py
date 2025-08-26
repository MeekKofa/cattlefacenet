"""
Simple timer utility for benchmarking training and evaluation.
"""

import time
from contextlib import contextmanager


class Timer:
    """Simple timer for measuring execution time."""

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        """Start the timer."""
        self.start_time = time.time()

    def stop(self):
        """Stop the timer and return elapsed time."""
        if self.start_time is None:
            raise ValueError("Timer not started")
        self.end_time = time.time()
        return self.elapsed()

    def elapsed(self):
        """Get elapsed time."""
        if self.start_time is None:
            return 0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

    @contextmanager
    def time_it(self):
        """Context manager for timing code blocks."""
        self.start()
        try:
            yield self
        finally:
            self.stop()

    def format_time(self, seconds=None):
        """Format time in human readable format."""
        if seconds is None:
            seconds = self.elapsed()

        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            seconds = seconds % 60
            return f"{int(minutes)}m {seconds:.1f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            seconds = seconds % 60
            return f"{int(hours)}h {int(minutes)}m {seconds:.0f}s"
