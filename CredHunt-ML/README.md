# Credential Generation System

A deep learning system for generating and analyzing credentials using PyTorch and Ray Tune.

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd credential-generation
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv\Scripts\activate  # On Windows: 
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Training and Generation

Run the system with default parameters:
```bash
python main.py
```

### Hyperparameter Tuning

Run with hyperparameter tuning:
```bash
python main.py --tune
```

## Project Structure

```
credential_generation/
│
├── src/
│   ├── __init__.py        # Package initialization
│   ├── dataset.py         # Dataset handling
│   ├── model.py           # Neural network model
│   ├── pipeline.py        # Training pipeline
│   └── tuning.py          # Hyperparameter tuning
│
├── main.py                # Main script
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Features

- Character-level credential generation
- LSTM-based neural network
- Hyperparameter tuning with Ray Tune
- GPU support (if available)
- Configurable generation parameters

## Hyperparameters

The following hyperparameters can be tuned:

- Model Architecture:
  - Embedding dimension
  - Hidden dimension
  - Number of LSTM layers
  - Dropout rate

- Training:
  - Batch size
  - Learning rate

## License

MIT License