# -*- coding: utf-8 -*-
"""Drop-in replacement for MessageAnnouncer that prints to stdout.

Use for CLI workflows where SSE streaming to a frontend is not needed.
Accepts the same `announce(msg)` calls as MessageAnnouncer so training_task()
and convert_model() can be invoked directly.
"""
import json
import sys
import time


class StdoutAnnouncer:
    def __init__(self, verbose=True, log_file=None):
        self.verbose = verbose
        self.log_file = log_file
        self._fh = open(log_file, "a", encoding="utf-8") if log_file else None

    def listen(self):
        raise NotImplementedError("StdoutAnnouncer does not support SSE listeners")

    def _fmt(self, msg):
        ts = msg.get("time", time.time())
        dt = time.strftime("%H:%M:%S", time.localtime(ts))
        event = msg.get("event", "?")
        body = msg.get("msg", "")
        extras = {k: v for k, v in msg.items() if k not in ("time", "event", "msg")}
        tail = f" | {json.dumps(extras, ensure_ascii=False)}" if extras else ""
        return f"[{dt}] [{event}] {body}{tail}"

    def announce(self, msg):
        line = self._fmt(msg)
        if self.verbose:
            print(line, flush=True)
        if self._fh:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None
