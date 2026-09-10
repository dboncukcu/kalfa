import numpy
import torch

from kalfa.registration import lego
from kalfa.std.feed.base import Dataset
from kalfa.std.pre.base import Tokenizer


class TokenDataset(Dataset):
    def __init__(self, frame, seq_len):
        self.frame = frame
        self.seq_len = int(seq_len)
        self.inputs = ["input_ids"]
        self.targets = ["targets"]
        chains = frame.chains.get("text", [])
        newline = None
        for preprocessor in chains:
            if isinstance(preprocessor, Tokenizer):
                newline = int(preprocessor.encode("\n")[0])
        pieces = []
        for position in range(len(frame.dataset)):
            value = frame.dataset[position]["text"]
            for preprocessor in chains:
                value = preprocessor.apply(value)
            pieces.append(numpy.asarray(value, dtype="int64").reshape(-1))
            if newline is not None:
                pieces.append(numpy.array([newline], dtype="int64"))
        stream = numpy.concatenate(pieces) if pieces else numpy.zeros(0, dtype="int64")
        count = max((len(stream) - 1) // self.seq_len, 0)
        starts = numpy.arange(count) * self.seq_len
        self.input_ids = torch.from_numpy(numpy.stack([stream[start:start + self.seq_len] for start in starts])) \
            if count else torch.zeros((0, self.seq_len), dtype=torch.int64)
        shifted = [stream[start + 1:start + self.seq_len + 1] for start in starts]
        self.target_ids = torch.from_numpy(numpy.stack(shifted)) \
            if count else torch.zeros((0, self.seq_len), dtype=torch.int64)
        self.starts = starts

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return {"input_ids": self.input_ids[index], "targets": self.target_ids[index]}

    def rows(self):
        return self.starts

    def labels(self, name):
        return self.target_ids


@lego("/feed/kalfa/next_token", alias="next_token",
      description="input_ids and targets windows of seq_len tokens over the set's token stream")
def next_token(frame, frames, seq_len):
    return TokenDataset(frame, seq_len)
