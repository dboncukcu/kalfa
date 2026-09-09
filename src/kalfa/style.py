"""Terminal colors: the palette the command line and the console log share."""

import os


class Style:
    def __init__(self, enabled):
        self.enabled = enabled

    def paint(self, text, code):
        return f"\x1b[{code}m{text}\x1b[0m" if self.enabled else text

    def bold(self, text):
        return self.paint(text, "1")

    def dim(self, text):
        return self.paint(text, "2")

    def red(self, text):
        return self.paint(text, "31")

    def green(self, text):
        return self.paint(text, "32")

    def yellow(self, text):
        return self.paint(text, "33")

    def cyan(self, text):
        return self.paint(text, "36")


def style_for(stream):
    is_tty = stream.isatty() if hasattr(stream, "isatty") else False
    return Style(is_tty and "NO_COLOR" not in os.environ)
