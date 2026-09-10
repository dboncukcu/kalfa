import hashlib
import os
import random

import numpy
import torch


def seed_all(seed):
    if seed is None:
        return
    random.seed(seed)
    numpy.random.seed(int(seed) % (2 ** 32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def derived_seed(seed, name):
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 63)


def rng_states():
    states = {"python": random.getstate(), "numpy": numpy.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states


def restore_rng(states):
    if not states:
        return
    if "python" in states:
        random.setstate(states["python"])
    if "numpy" in states:
        numpy.random.set_state(states["numpy"])
    if "torch" in states:
        torch.set_rng_state(states["torch"])
    if states.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["cuda"])


class Seeded:
    def __init__(self, context, seed):
        self.context = context
        self.seed = seed

    def __enter__(self):
        self.context.__enter__()
        torch.manual_seed(self.seed)
        return self

    def __exit__(self, *error):
        return self.context.__exit__(*error)


def forked(seed):
    if seed is None:
        return torch.random.fork_rng(devices=[], enabled=False)
    return Seeded(torch.random.fork_rng(devices=[]), seed)


def process_seed():
    info = torch.utils.data.get_worker_info()
    return int(info.seed if info is not None else torch.initial_seed())


def seed_worker(worker_id):
    seed = process_seed() % (2 ** 32)
    random.seed(seed)
    numpy.random.seed(seed)


class Draws:
    def __init__(self, salt):
        self.salt = int.from_bytes(hashlib.sha256(str(salt).encode()).digest()[:4], "big")
        self.process = None
        self.generator = None

    def numpy(self):
        process = os.getpid()
        if self.generator is None or self.process != process:
            self.process = process
            self.generator = numpy.random.default_rng([process_seed(), self.salt])
        return self.generator
