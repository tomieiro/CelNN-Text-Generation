"""Small byte-level BPE tokenizer with an explicit chat protocol."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = (
    "<pad>",
    "<unk>",
    "<bos>",
    "<user>",
    "<assistant>",
    "<eos>",
)


class ChatTokenizer:
    """Serializable BPE tokenizer whose role IDs are stable and inspectable."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer
        missing = [token for token in SPECIAL_TOKENS if self.id(token) is None]
        if missing:
            raise ValueError(f"tokenizer lacks special tokens: {missing}")

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        *,
        vocab_size: int = 1_024,
        min_frequency: int = 2,
    ) -> "ChatTokenizer":
        """Train byte-level BPE without an unknown Unicode character class."""
        if vocab_size < len(SPECIAL_TOKENS) + 256:
            raise ValueError("vocab_size must fit byte alphabet and specials")
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.normalizer = NFKC()
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=list(SPECIAL_TOKENS),
            initial_alphabet=ByteLevel.alphabet(),
        )
        tokenizer.train_from_iterator(texts, trainer=trainer)
        return cls(tokenizer)

    @classmethod
    def load(cls, path: str | Path) -> "ChatTokenizer":
        return cls(Tokenizer.from_file(str(path)))

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(destination))

    def id(self, token: str) -> int | None:
        return self._tokenizer.token_to_id(token)

    def special_id(self, token: str) -> int:
        if token not in SPECIAL_TOKENS:
            raise ValueError(f"not a chat special token: {token}")
        token_id = self.id(token)
        assert token_id is not None
        return token_id

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    @property
    def pad_id(self) -> int:
        return self.special_id("<pad>")

    @property
    def eos_id(self) -> int:
        return self.special_id("<eos>")

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids: Iterable[int], *, skip_special: bool = True) -> str:
        return self._tokenizer.decode(
            list(ids), skip_special_tokens=skip_special
        )
