# torch_checkpointing

A high-performance, distributed-training-ready checkpointing library for PyTorch models.

# Under Construction!

Core functionality is implemented but APIs are subject to change. Check back soon for stable release.

## Installation

### Basic Installation
```bash
# Clone the repository
git clone https://github.com/your-org/torch_checkpointing.git
cd torch_checkpointing

# Install basic dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Development Setup
For contributors and developers, install all development dependencies:

```bash
# Clone the repository
git clone https://github.com/your-org/torch_checkpointing.git
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
from torch_checkpointing import make_sync_checkpointer, make_async_checkpointer
```

## Usage

### Basic Synchronous Checkpointing

```python
import torch
from torch_checkpointing import make_sync_checkpointer

# Create a model
model = torch.nn.Linear(10, 1)
state_dict = model.state_dict()

# Create a synchronous checkpointer
checkpointer = make_sync_checkpointer()

# Save checkpoint
checkpointer.save("model_checkpoint.pt", state_dict)

# Load checkpoint
loaded_state_dict = checkpointer.load("model_checkpoint.pt")
model.load_state_dict(loaded_state_dict)

# Cleanup
checkpointer.close()
```

### Asynchronous Checkpointing for Performance

```python
import torch
from torch_checkpointing import make_async_checkpointer

async def train_with_async_checkpointing():
    model = torch.nn.Linear(10, 1)
    checkpointer = make_async_checkpointer()

    for epoch in range(100):
        # ... training code ...

        # Save checkpoint asynchronously
        stage_future, write_future = checkpointer.save(
            f"checkpoint_epoch_{epoch}.pt",
            model.state_dict()
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
- **Hook System**: Extensible hooks for custom checkpoint processing
- **Multiple Formats**: Support for PyTorch and SafeTensors formats
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

4. **Test thoroughly** -- All features must have unit tests and integration tests for distributed scenarios.
