from pathlib import Path

import numpy


HERE = Path(__file__).resolve().parent
LINES = 150
SPEAKERS = ("ROMEO", "JULIET", "NURSE", "MERCUTIO", "FRIAR")
OPENERS = ("o", "but", "and", "now", "come", "good", "sweet", "alas", "soft", "nay")
ADJECTIVES = ("gentle", "fair", "sweet", "dark", "bright", "sad", "young", "true", "cruel", "soft", "pale", "wild")
NOUNS = ("night", "love", "moon", "light", "heart", "name", "rose", "sword", "day", "star", "window", "house",
         "kiss", "death", "dream", "lady", "tomb", "morning")
VERBS = ("speaks", "breaks", "burns", "sleeps", "dies", "shines", "waits", "calls", "weeps", "flies", "fades",
         "sings")
ENDS = (".", ".", "?", "!")


def clause(generator):
    return f"the {generator.choice(ADJECTIVES)} {generator.choice(NOUNS)} {generator.choice(VERBS)}"


def line(generator):
    words = f"{generator.choice(OPENERS)}, {clause(generator)}"
    if generator.random() < 0.6:
        words += f", and {clause(generator)}"
    return f"{generator.choice(SPEAKERS)}: {words}{generator.choice(ENDS)}"


def corpus(lines=LINES, seed=5):
    generator = numpy.random.default_rng(seed)
    return "\n".join(line(generator) for _ in range(lines)) + "\n"


def main():
    path = HERE / "data" / "corpus.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = corpus()
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path} ({LINES} lines, {len(text)} characters, {len(set(text))} distinct)")


if __name__ == "__main__":
    main()
