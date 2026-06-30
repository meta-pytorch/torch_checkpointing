# torch_checkpointing

A high-performance, distributed-training-ready checkpointing library for PyTorch models.

# Under Construction!

Core functionality is implemented but APIs are subject to change. Check back soon for stable release.

### Development Setup
For contributors and developers, install all development dependencies:

```bash
# Clone the repository
git clone https://github.com/meta-pytorch/torch_checkpointing.git
cd torch_checkpointing

# Create conda environment
conda create -n torch_checkpointing python=3.12
conda activate torch_checkpointing

# Install all dependencies (includes linting, testing, type checking)
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Development Tools
The requirements include:
- **PyTorch 2.6+**: Core functionality
- **pytest**: Testing framework with timeout support
- **mypy**: Static type checking
- **flake8 & ruff**: Code linting and formatting
- **pygls**: Language server for IDE support

Once installed, you can import it in your Python code:

```python
from torch_checkpointing import make_sync_checkpoint_saver, make_async_checkpoint_saver
```

## Usage

### Basic Synchronous Checkpointing

```python
import torch
from torch_checkpointing import (
    CheckpointLoader,
    CheckpointReader,
    make_sync_checkpoint_saver,
    RankInfo,
)
from torch_checkpointing.checkpoint_base import CheckpointBase, CheckpointItem
from torch_checkpointing.storage.filesystem import LocalFileSystemStorageConfig


# A minimal CheckpointBase subclass that wraps a model's state dict.
class SimpleCheckpoint(CheckpointBase):
    def __init__(self, state_dict):
        self.state_dict = state_dict

    def get_items(self) -> dict[str, CheckpointItem]:
        return {k: CheckpointItem(value=v) for k, v in self.state_dict.items()}

    def load_state_dict(self, state_dict) -> None:
        self.state_dict = state_dict


# Create a model
model = torch.nn.Linear(10, 1)
state_dict = model.state_dict()

# Create a synchronous checkpointer
checkpointer = make_sync_checkpoint_saver()

# Create checkpoint and save
checkpoint = SimpleCheckpoint(state_dict)  # SimpleCheckpoint is a CheckpointBase subclass
checkpointer.save("model_checkpoint.pt", checkpoint)

# Load the checkpoint with a CheckpointLoader
reader = CheckpointReader(
    rank_info=RankInfo(global_world_size=1, global_rank=0),
    storage_config=LocalFileSystemStorageConfig(),
)
loader = CheckpointLoader(reader=reader)
loaded_checkpoint = SimpleCheckpoint(state_dict)
loader.load("model_checkpoint.pt", loaded_checkpoint)
loader.close()
# loaded_checkpoint is now updated in-place
model.load_state_dict(loaded_checkpoint.state_dict)

# Cleanup
checkpointer.close()
```

### Asynchronous Checkpointing for Performance

```python
import torch
from torch_checkpointing import make_async_checkpoint_saver
from torch_checkpointing.checkpoint_base import CheckpointBase

async def train_with_async_checkpointing():
    model = torch.nn.Linear(10, 1)
    checkpointer = make_async_checkpoint_saver()

    for epoch in range(100):
        # ... training code ...

        # Create checkpoint and save asynchronously
        checkpoint = SimpleCheckpoint(model.state_dict())  # SimpleCheckpoint is a CheckpointBase subclass
        stage_future, write_future = checkpointer.save(
            f"checkpoint_epoch_{epoch}.pt",
            checkpoint
        )

        # Continue training while checkpoint is being written
        # ... more training code ...

        # Optionally wait for completion when needed
        if epoch % 10 == 0:
            write_future.result()  # Ensure checkpoint is complete

    checkpointer.close()
```

### Distributed Training Support
Yet to be implemented

## Key Features

- **Async Checkpointing**: Non-blocking checkpoint operations for minimal training interruption
- **Distributed Training**: Built-in synchronization barriers for multi-process training
- **Flexible Architecture**: Modular design with customizable readers, writers, and stagers
- **Customizable Callbacks**: Pre- and post-finalize callbacks for custom checkpoint logic
- **Pluggable Storage**: Filesystem storage backend with a pluggable `StorageConfig` interface
- **Performance Optimized**: Efficient staging and background writing

## Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

Run specific test modules:

```bash
python -m pytest tests/test_checkpointer.py -v
python -m pytest tests/test_barriers.py -v
```

## Development Tools

### Code Quality & Linting
```bash
# Format code with ruff
ruff format .

# Lint code with ruff
ruff check .

# Lint with flake8
flake8 torch_checkpointing tests

# Type checking with mypy
mypy torch_checkpointing
```

### Running All Checks
```bash
# Run all quality checks at once
ruff check . && flake8 torch_checkpointing tests && mypy torch_checkpointing && pytest tests/
```

# Contributing Guidelines

1. **Build iteratively** -- Develop features incrementally with clear use cases and comprehensive tests.

2. **Maintain backward compatibility** -- API changes should be carefully considered and well-documented.

3. **Performance first** -- Checkpointing should minimize impact on training performance.

4. **Test thoroughly** -- All features must have unit tests.
