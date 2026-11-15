import logging
import threading
from collections import deque


KEPT = 256
LOGGER = logging.getLogger("kalfa.board")


class Watcher:
    def __init__(self, board, every=2.0):
        self.board = board
        self.every = float(every)
        self.condition = threading.Condition()
        self.version = 0
        self.changed = []
        self.stamps = {}
        self.thread = None
        self.stopping = False
        self.log = deque(maxlen=KEPT)
        self.failure = None

    def start(self):
        with self.condition:
            if self.thread is not None and self.thread.is_alive():
                return self
            self.stopping = False
            found = self.read()
            if found is not None:
                self.stamps = found
            self.thread = threading.Thread(target=self.run, name="kalfa-watch", daemon=True)
            self.thread.start()
        return self

    def stop(self):
        with self.condition:
            self.stopping = True
            self.condition.notify_all()
            thread, self.thread = self.thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join()

    def wait(self, version, timeout):
        with self.condition:
            self.condition.wait_for(lambda: self.version > version or self.stopping, timeout)
            return self.version if self.version > version else None

    def since(self, version):
        with self.condition:
            changed = {path for number, paths in self.log if number > version for path in paths}
            missed = self.version > version and (not self.log or self.log[0][0] > version + 1)
            if missed:
                changed.update(self.stamps)
            return self.version, sorted(changed)

    def run(self):
        while True:
            with self.condition:
                if self.condition.wait_for(lambda: self.stopping, self.every):
                    return
            self.poll()

    def poll(self):
        found = self.read()
        if found is None or found == self.stamps:
            return False
        with self.condition:
            changed = sorted(path for path in found.keys() | self.stamps.keys()
                             if found.get(path) != self.stamps.get(path))
            self.stamps = found
            self.changed = changed
            self.version += 1
            self.log.append((self.version, changed))
            self.condition.notify_all()
        return True

    def read(self):
        try:
            found = dict(self.board.stamps())
        except Exception as error:
            text = f"{type(error).__name__}: {error}"
            if text != self.failure:
                LOGGER.warning("the board could not read its stamps: %s", text)
            self.failure = text
            return None
        self.failure = None
        return found
