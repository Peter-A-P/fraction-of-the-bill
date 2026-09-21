"""Fine-tuning: the dataset, the recipe, the checkpoints and the run.

Importing this package costs nothing on a machine with no GPU stack. Everything that
needs torch is behind a function call in `qlora`, so `smallprint data build` and the
tests run on the laptop while the training runs on rented hardware.
"""

from smallprint.train.checkpoint import CheckpointStore, LocalSyncer
from smallprint.train.dataset import Example, build, read, write
from smallprint.train.recipe import TrainConfig, seeds, sweep

__all__ = [
    "CheckpointStore",
    "Example",
    "LocalSyncer",
    "TrainConfig",
    "build",
    "read",
    "seeds",
    "sweep",
    "write",
]
