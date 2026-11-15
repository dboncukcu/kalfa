from dataclasses import dataclass


class KalfaError(Exception):
    pass


class ConfigError(KalfaError):
    pass


class BuildError(KalfaError):
    pass


class RunError(KalfaError):
    pass


@dataclass
class Problem:
    code: str
    message: str
    path: str | None = None
    hint: str | None = None
    severity: str = "error"

    def __str__(self):
        where = f"{self.path}: " if self.path else ""
        tail = f" ({self.hint})" if self.hint else ""
        return f"{self.severity} {self.code}: {where}{self.message}{tail}"


def error(code, message, path=None, hint=None):
    return Problem(code, message, path, hint, "error")


def warning(code, message, path=None, hint=None):
    return Problem(code, message, path, hint, "warning")


class ProblemsError(KalfaError):
    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("\n".join(str(problem) for problem in self.problems))
