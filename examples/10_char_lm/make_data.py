"""Data of example 10: data/shakespeare.txt, two thousand short generated lines, a ROMEO line every fifth."""

from pathlib import Path

import numpy

HERE = Path(__file__).parent


def write_text(path, lines=200, seed=0):
    generator = numpy.random.default_rng(seed)
    words = ["love", "night", "sun", "moon", "sword", "fair", "verona", "friend", "death", "light"]
    rows = []
    for position in range(lines):
        count = int(generator.integers(3, 8))
        text = " ".join(generator.choice(words, size=count))
        rows.append(f"ROMEO: {text}" if position % 5 == 0 else text)
    Path(path).write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def main():
    (HERE / "data").mkdir(exist_ok=True)
    write_text(HERE / "data" / "shakespeare.txt", lines=2000)
    print("wrote data/shakespeare.txt (2000 lines)")


if __name__ == "__main__":
    main()
