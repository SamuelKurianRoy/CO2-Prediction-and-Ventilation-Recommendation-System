# CO₂-based Ventilation Recommendation System

This project implements a deep learning-based CO₂ prediction and ventilation recommendation system based on research published in Energy Reports (2025). The system uses three different models (TCN, SimpleRNN, and Sequence-to-Sequence) to predict indoor CO₂ concentrations and provide ventilation recommendations.

## Requirements

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for faster training)

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python co2_prediction.py
```

The script will:
1. Generate synthetic CO₂ data
2. Train three different models (TCN, SimpleRNN, S2S)
3. Evaluate model performance
4. Generate predictions and ventilation recommendations
5. Create visualization plots

## Output

The script generates:
- Trained model files (.h5)
- Performance metrics for each model
- Visualization plots (.png)
- Predictions and recommendations (CSV)

## Model Architecture

- TCN: 64 filters, kernel size 3, dilations [1, 2, 4, 8]
- SimpleRNN: 50 units
- Sequence-to-Sequence: LSTM-based encoder-decoder with 50 units

## Performance

Expected performance based on the paper:
- TCN: RMSE = 50.46 ppm, R² = 0.97
- SimpleRNN: RMSE = 41.97 ppm, R² = 0.98

## License

MIT License 