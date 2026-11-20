"""Serialize a pending vendor acquisition, acknowledged exit, and optional re-entry."""
import time


class ReleaseGate:
    def __init__(self):
        self.pending = False
        self.reenter = False
        self.prior = self.release = None
        self.error = ""
        self.failed = False

    def begin(self, acquisition, timeout, now=None):
        self.reenter = False
        if self.pending:
            return
        self.pending = True
        self.prior = acquisition
        self.release = None
        self.error = ""
        self.failed = False
        self.deadline = (time.monotonic() if now is None else now) + timeout

    def tick(self, send_release, now=None):
        if not self.pending or self.failed:
            return False
        now = time.monotonic() if now is None else now
        # Never start another session while a late acquisition can still succeed.
        if now > self.deadline:
            self.error = "SYNC exit acknowledgement delayed"
            self.reenter = False
        if self.prior is not None and not self.prior.done():
            return False
        self.prior = None
        try:
            if self.release is None:
                self.release = send_release()
                return False
            if not self.release.done():
                return False
            result = self.release.result()
            if result is None or not result.success:
                raise RuntimeError(getattr(result, "message", "empty exit reply"))
        except Exception as exc:
            self.error = str(exc)
            self.failed = True
            self.reenter = False
            return False
        resume = self.reenter
        self.pending = self.reenter = False
        self.error = ""
        return resume

    def retry(self, timeout):
        if self.failed:
            self.pending = False
            self.begin(None, timeout)
