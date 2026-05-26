# IRIS FLOWER CLASSIFICATION

## Overview
The Iris flower dataset consists of three species: **Setosa**, **Versicolor**, and **Virginica**. These species can be distinguished based on their measurements (sepal length, sepal width, petal length, petal width). This project develops a machine learning model that can accurately classify Iris flowers into their respective species.

## Dataset
- **Total Samples**: 150 Iris flowers
- **Features**: 4 numerical features
  - Sepal Length (cm)
  - Sepal Width (cm)
  - Petal Length (cm)
  - Petal Width (cm)
- **Classes**: 3 species (Setosa, Versicolor, Virginica)
- **Source**: The Iris dataset is included in scikit-learn

## Project Structure
```
Task 3/
├── README.md                          # This file
├── iris_flower_classification.py      # Main Python script
└── iris_model.pkl                     # Trained model (generated after running)
```

## What Was Covered

### 1. **Exploratory Data Analysis (EDA)**
   - Dataset overview and basic statistics
   - Feature distributions (histograms)
   - Correlation analysis between features
   - Class distribution visualization
   - Pairplot to visualize relationships between features

### 2. **Data Preprocessing**
   - Handling missing values (if any)
   - Feature standardization using StandardScaler
   - Train-test split (80-20 ratio)

### 3. **Machine Learning Models**
   Multiple classification models trained and compared:
   - **Logistic Regression** - Baseline linear classifier
   - **Decision Tree Classifier** - Tree-based approach
   - **Random Forest Classifier** - Ensemble method
   - **Support Vector Machine (SVM)** - Advanced classifier
   - **K-Nearest Neighbors (KNN)** - Instance-based learning

### 4. **Model Evaluation**
   - **Accuracy Score**: Overall correctness
   - **Precision, Recall, F1-Score**: Per-class performance
   - **Confusion Matrix**: Classification breakdown
   - **Classification Report**: Comprehensive metrics
   - **Cross-Validation Score**: Model stability

### 5. **Visualizations**
   - Feature distributions
   - Correlation heatmap
   - Class distribution
   - Confusion matrices for each model
   - Model performance comparison

## Requirements
```
pandas==2.0.0
numpy==1.24.0
matplotlib==3.7.0
seaborn==0.12.0
scikit-learn==1.2.0
```

## Installation & Setup

1. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```
   Or install individually:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

2. **Run the script**:
   ```bash
   python iris_flower_classification.py
   ```

## Output
The script will:
- Display model performance metrics in the console
- Generate visualization plots (displayed during execution)
- Save the trained Random Forest model as `iris_model.pkl`
- Print feature importance and predictions

## Key Insights
- **Best Performing Model**: Random Forest typically achieves ~97% accuracy
- **Most Important Features**: Petal length and petal width are usually the most discriminative
- **Class Balance**: All three species have 50 samples each (balanced dataset)
- **Linear Separability**: The dataset is relatively easy to classify

## Usage Example
```python
from sklearn.externals import joblib

# Load the trained model
model = joblib.load('iris_model.pkl')

# Make predictions
prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])
```

## Video Demo
**Please record a video demonstration of your application and upload it to LinkedIn. Then submit the LinkedIn URL to complete this task.**

The video should showcase:
- Loading and exploring the dataset
- Training the model(s)
- Model evaluation results
- Making predictions on new data
- Discussing the findings and accuracy

## Future Enhancements
- Hyperparameter tuning using GridSearchCV
- Deep learning approaches (Neural Networks)
- Deployment as a web application (Flask/Streamlit)
- Real-time prediction interface
- Feature importance analysis visualization

## References
- [Iris Dataset Documentation](https://en.wikipedia.org/wiki/Iris_flower_data_set)
- [Scikit-learn Classification Guide](https://scikit-learn.org/stable/modules/classification.html)

---
**Author**: Your Name  
**Date**: May 2026  
**Status**: Completed
