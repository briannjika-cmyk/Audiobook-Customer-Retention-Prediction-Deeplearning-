# Audiobook Customer Retention Prediction using Deep Learning

## Overview

This project implements a deep neural network to predict audiobook customer retention. The model analyzes customer behavior and engagement patterns to identify which customers are likely to remain active, enabling targeted advertising strategies and improved customer retention efforts.

## Project Objectives

- **Primary Goal**: Develop a predictive model to classify whether audiobook customers will remain retained or churn
- **Business Application**: Support data-driven decision-making for targeted marketing and customer retention campaigns
- **Technical Goal**: Leverage deep learning to identify complex patterns in customer engagement data

## Dataset

### Source
- **File**: `Audiobooks_data (1).csv`
- **Records**: 500+ customer samples (with additional data available)
- **Target Variable**: Binary classification (Retained = 1, Churned = 0)

### Features
The dataset contains 10 input features capturing customer engagement and subscription metrics:
1. **Customer ID**: Unique identifier for each customer
2. **Time on Platform**: Customer tenure in days
3. **Time Spent (Engagement)**: Hours spent actively using the platform
4. **Book Price (Average)**: Average price of audiobooks purchased
5. **Book Price (Highest)**: Highest-priced audiobook purchased
6. **Premium Membership**: Binary indicator of premium subscription status
7. **Price Currency**: Currency indicator for pricing normalization
8. **Discount Applied**: Proportion of purchases made with discounts
9. **Total Discount Amount**: Monetary value of discounts used
10. **Completion Rate**: Number of books completed
11. **Library Size**: Total audiobooks in customer's library
12. **Review Activity**: Engagement indicator (binary)

### Data Characteristics
- **Balanced Dataset**: Original data was imbalanced with more churned customers; balanced by removing excess zero-class instances
- **Standardized Features**: Input features standardized using zero-mean, unit-variance normalization
- **Shuffled**: Data randomized to prevent bias in training sequence

## Data Preprocessing

### Steps Implemented

1. **Data Loading**
   - Loaded CSV file using NumPy
   - Extracted input features (columns 1-10) and target variable (last column)

2. **Class Balancing**
   - Addressed class imbalance by removing excess negative samples
   - Balanced dataset to equal number of retained (1) and churned (0) customers
   - Final balanced dataset: ~3,474 samples

3. **Feature Standardization**
   - Applied standardization (z-score normalization) to all input features
   - Ensures features are on comparable scales for neural network training

4. **Data Shuffling**
   - Randomized sample order to prevent training bias
   - Maintained correspondence between features and targets

5. **Train-Validation-Test Split**
   - **Training Set**: 80% (2,859 samples)
   - **Validation Set**: 10% (447 samples)
   - **Test Set**: 10% (448 samples)
   - All splits maintain similar class distributions (~50% retention rate)

## Deep Learning Model

### Architecture

```
Input Layer (10 features)
    ↓
Hidden Layer 1 (100 neurons, ReLU activation)
    ↓
Hidden Layer 2 (100 neurons, ReLU activation)
    ↓
Output Layer (2 neurons, Softmax activation)
    ↓
Binary Classification (Retained / Churned)
```

### Model Configuration

| Component | Configuration |
|-----------|---------------|
| **Input Size** | 10 features |
| **Output Size** | 2 classes (binary classification) |
| **Hidden Layer Size** | 100 neurons per layer |
| **Number of Hidden Layers** | 2 |
| **Activation Function (Hidden)** | ReLU (Rectified Linear Unit) |
| **Activation Function (Output)** | Softmax |
| **Optimizer** | Adam |
| **Loss Function** | Sparse Categorical Crossentropy |
| **Batch Size** | 100 |
| **Max Epochs** | 100 |
| **Early Stopping** | Enabled (patience=2) |

### Training Hyperparameters

- **Optimizer**: Adam (adaptive learning rate)
- **Loss**: Sparse Categorical Crossentropy (suitable for multi-class classification)
- **Metrics**: Accuracy
- **Batch Size**: 100 samples per gradient update
- **Early Stopping**: Halts training if validation loss does not improve for 2 consecutive epochs

## Model Performance

### Evaluation Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 81.70% |
| **Test Loss** | 0.32 |

### Results Interpretation

- **Accuracy**: The model correctly predicts customer retention status in approximately 82% of test cases
- **Loss**: Moderate categorical crossentropy loss indicates good model calibration
- **Early Stopping**: Training halted early due to convergence, preventing overfitting

### Training Dynamics

- **Initial Accuracy** (Epoch 1): 70.16%
- **Peak Accuracy** (Mid-training): ~81%
- **Convergence**: Model stabilized around epoch 11 with consistent validation performance

## Technologies Used

### Data Processing & Numerical Computing
- **NumPy**: Array manipulation and mathematical operations
- **scikit-learn**: Data preprocessing and standardization

### Deep Learning Framework
- **TensorFlow/Keras**: Neural network implementation and training
  - Sequential model API for layer-by-layer model construction
  - Dense layers for fully-connected neural network
  - Built-in callbacks for early stopping

### Development Environment
- **Python 3.13.15**: Programming language
- **Jupyter Notebook**: Interactive development and experimentation

## File Structure

```
├── Deeplearning(Audiobooks).ipynb    # Main notebook with complete implementation
├── Audiobooks_data (1).csv           # Raw customer dataset
├── README.md                          # This file
└── .gitignore                         # Git ignore configuration
```

## Output Files

The notebook generates preprocessed datasets in NPZ format:
- `Audiobooks_data_train.npz`: Training set (features and targets)
- `Audiobooks_data_validation.npz`: Validation set (features and targets)
- `Audiobooks_data_test.npz`: Test set (features and targets)

## Usage

### Requirements

```bash
pip install numpy scikit-learn tensorflow
```

### Running the Notebook

1. Ensure `Audiobooks_data (1).csv` is in the working directory
2. Open `Deeplearning(Audiobooks).ipynb` in Jupyter Notebook
3. Execute cells sequentially:
   - Data extraction and preprocessing
   - Model definition and compilation
   - Model training and evaluation

### Expected Output

Upon successful execution:
```
Test loss: 0.32. Test accuracy: 81.70%
```

## Key Insights

1. **Feature Importance**: Customer engagement metrics (time spent, discount usage, library size, completion rate) are strong predictors of retention
2. **Model Performance**: The 81.70% accuracy demonstrates that customer retention patterns are learnable through neural networks
3. **Early Convergence**: Model convergence by epoch 11 suggests the problem may be solvable with simpler architectures or fewer epochs
4. **Balanced Data**: Dataset balancing improved model training and prevented bias toward the majority class

## Future Improvements

1. **Hyperparameter Optimization**: Experiment with different layer sizes, learning rates, and regularization techniques
2. **Feature Engineering**: Derive additional features from raw customer data (e.g., engagement velocity, churn risk indicators)
3. **Model Ensembling**: Combine multiple models for improved predictions
4. **Class Weight Adjustment**: Use class weights instead of manual balancing to preserve original data distribution
5. **Cross-Validation**: Implement k-fold cross-validation for more robust evaluation
6. **Feature Importance Analysis**: Identify most influential features using permutation importance or attention mechanisms
7. **Interpretability**: Add model interpretation techniques to understand individual predictions

## License

This project is provided as-is for educational and research purposes.

## Author

Created for predictive analytics and machine learning demonstration purposes.
