import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from hw1_imitation import train


class TinyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        assert self.training
        return (self.weight - action_chunk.mean()).square()


class TestTrainingLoop(unittest.TestCase):
    def test_executes_configured_schedule(self) -> None:
        model = TinyPolicy()
        logger = mock.Mock()
        batch = (torch.zeros(1, 1), torch.ones(1, 1, 1))
        evaluate = mock.Mock(side_effect=lambda **kwargs: kwargs["model"].eval())

        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            with (
                mock.patch.object(
                    train, "download_pusht", return_value=tmp_path / "data"
                ),
                mock.patch.object(
                    train,
                    "load_pusht_zarr",
                    return_value=(
                        np.zeros((1, 1), dtype=np.float32),
                        np.zeros((1, 1), dtype=np.float32),
                        np.ones(1, dtype=np.int64),
                    ),
                ),
                mock.patch.object(train, "PushtChunkDataset"),
                mock.patch.object(
                    train, "DataLoader", return_value=[batch, batch, batch]
                ),
                mock.patch.object(train, "build_policy", return_value=model),
                mock.patch.object(train, "Logger", return_value=logger),
                mock.patch.object(train, "evaluate_policy", evaluate, create=True),
                mock.patch.object(train.wandb, "init"),
                mock.patch.object(train, "LOGDIR_PREFIX", tmp_path),
            ):
                train.run_training(
                    train.TrainConfig(
                        num_epochs=1,
                        log_interval=1,
                        eval_interval=2,
                        batch_size=1,
                    )
                )

        self.assertNotEqual(model.weight.item(), 0.0)
        self.assertEqual(
            [call.kwargs["step"] for call in logger.log.call_args_list], [1, 2, 3]
        )
        self.assertEqual(evaluate.call_args.kwargs["step"], 2)
        logger.dump_for_grading.assert_called_once_with()
