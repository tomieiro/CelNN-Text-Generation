import numpy as np

from celllm.config import ModelConfig, TrainConfig
from celllm.data import Batcher
from celllm.model import CelNNLanguageModel
from celllm.train import evaluate, set_seed, train


def _repeating_batcher(seed=0):
    ids = np.tile(np.array([1, 2, 3], dtype=np.int64), 20_000)
    return Batcher(ids, n=64, batch_size=16, seed=seed)


def test_evaluate_returns_bpc_near_uniform_for_untrained_model():
    set_seed(0)
    cfg = ModelConfig(n=64, d=32, vocab_size=27, mixer="rank8")
    bpc = evaluate(CelNNLanguageModel(cfg), _repeating_batcher(), n_batches=3)
    assert 4.0 < bpc < 5.5


def test_training_reduces_bpc_on_a_learnable_stream():
    set_seed(0)
    cfg = ModelConfig(n=64, d=64, vocab_size=27, mixer="rank8")
    model = CelNNLanguageModel(cfg)
    train_config = TrainConfig(
        steps=300,
        batch_size=16,
        lr=3e-3,
        warmup=20,
        eval_every=150,
        eval_batches=3,
    )
    result = train(
        model, _repeating_batcher(), _repeating_batcher(1), train_config
    )
    assert result["final_bpc"] < 3.0, result["history"]


def test_set_seed_makes_training_reproducible():
    def run():
        set_seed(11)
        cfg = ModelConfig(n=64, d=32, vocab_size=27, mixer="rank4")
        model = CelNNLanguageModel(cfg)
        train_config = TrainConfig(
            steps=20,
            batch_size=8,
            warmup=5,
            eval_every=20,
            eval_batches=2,
        )
        return train(
            model, _repeating_batcher(), _repeating_batcher(1), train_config
        )

    assert run()["final_bpc"] == run()["final_bpc"]
