# SignEval - ASL Alphabet Recognition

SignEval is a real-time American Sign Language (ASL) alphabet recognition system that uses hand landmarks to identify sign language gestures through a webcam.

## Features

- Real-time ASL alphabet sign detection
- Uses MediaPipe for hand landmark detection
- Deep learning model for accurate sign classification
- Interactive webcam interface with feedback
- Support for all 26 ASL alphabet letters plus "space", "delete", and "nothing" gestures

## Project Structure

```
signeval/
├── data/
│   └── preprocessing.py    # Data preprocessing utilities
├── models/
│   ├── train.py            # Model training script
│   └── evaluate.py         # Model evaluation utilities
├── utils/
│   └── landmark_utils.py   # Hand landmark extraction utilities
├── app.py                  # Main application script
├── config.py               # Configuration parameters
├── requirements.txt        # Required Python packages
└── README.md               # Project documentation
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/signeval.git
cd signeval
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
pip install -r requirements.txt
```

3. Prepare your dataset:
   - Ensure the ASL alphabet dataset is placed in `dataset/asl_alphabet_train` and `dataset/asl_alphabet_test`
   - The dataset should contain 29 folders (A-Z, del, nothing, space)

## Usage

### 1. Preprocess the data

```bash
python data/preprocessing.py
```

This will:
- Extract hand landmarks from images
- Normalize the landmarks
- Split the data into training and validation sets
- Save the processed data to `data/processed/`

### 2. Train the model

```bash
python models/train.py
```

This will:
- Load the preprocessed data
- Train a deep learning model
- Save the trained model to `saved_models/`
- Generate training history plots

### 3. Evaluate the model

```bash
python models/evaluate.py
```

This will:
- Load the latest trained model
- Evaluate its performance on the validation set
- Generate a confusion matrix and classification report
- Analyze common misclassifications

### 4. Run the application

```bash
python app.py
```

This will:
- Load the trained model
- Open your webcam
- Start real-time ASL sign recognition

## Customization

You can modify the configuration parameters in `config.py` to adjust:
- Dataset paths
- Model architecture
- Training parameters
- Real-time prediction settings

## Model Details

- **Input**: 63-dimensional vector (21 hand landmarks with x, y, z coordinates)
- **Architecture**: Dense neural network with batch normalization and dropout
- **Output**: 29 classes (A-Z, del, nothing, space)

## License

[License Information]

## Acknowledgments

- The ASL alphabet dataset
- MediaPipe for hand landmark detection
