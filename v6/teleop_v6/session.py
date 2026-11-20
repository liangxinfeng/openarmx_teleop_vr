"""One ordered IK worker; replace waiting poses and cancel work at control boundaries."""
import threading


class SessionMailbox:
    def __init__(self):
        self._condition = threading.Condition()
        self._generation = 0
        self._sequence = 0
        self._pending = None
        self._closed = False
        self.replaced_pending = 0
        self.discarded_inflight = 0

    def put(self, value):
        with self._condition:
            if self._closed:
                return
            self._sequence += 1
            self.replaced_pending += int(self._pending is not None)
            ticket = self._generation, self._sequence
            self._pending = ticket, value
            self._condition.notify()
            return ticket

    def take(self):
        with self._condition:
            self._condition.wait_for(lambda: self._closed or self._pending is not None)
            sample, self._pending = self._pending, None
            return sample

    def is_latest(self, ticket):
        """Compatibility with the worker: one in-flight solve per generation."""
        with self._condition:
            return not self._closed and ticket[0] == self._generation

    def reset(self):
        with self._condition:
            self._generation += 1
            self._pending = None

    def close(self):
        with self._condition:
            self._closed = True
            self._pending = None
            self._condition.notify_all()

    def discard_inflight(self):
        with self._condition:
            self.discarded_inflight += 1

    def stats(self):
        with self._condition:
            return dict(generation=self._generation, latest_sequence=self._sequence,
                        pending=self._pending is not None, replaced_pending=self.replaced_pending,
                        discarded_inflight=self.discarded_inflight)
