# Deep Learning Based Approach for Exoplanet Detection

This project develops a deep learning model to analyze time-series data of stellar flux values to identify exoplanet transits. The approach utilizes various deep learning techniques and data augmentation methods to enhance the accuracy of exoplanet detection.

## Project Overview

The goal of this project is to leverage deep learning models to identify exoplanets based on the periodic dimming of stars caused by planetary transits. The dataset used is based on the transit method, extensively utilized by the Kepler Mission, which monitors stellar brightness to detect these transits.

## Dataset

The dataset comprises flux values recorded over time for several thousand stars, with each star labeled as either having an exoplanet (2) or not (1).

- **Trainset:**
  - Rows: 5087
  - Features: 3198
  - Exoplanet-stars: 37
  - Non-exoplanet-stars: 5050

- **Testset:**
  - Rows: 570
  - Features: 3198
  - Exoplanet-stars: 5
  - Non-exoplanet-stars: 565

## Techniques Used

### Data Imbalance Handling
- **SMOTE (Synthetic Minority Over-sampling Technique):** Creates synthetic samples for the minority class.
- **ADASYN (Adaptive Synthetic Sampling):** Creates synthetic minority class examples, focusing on harder-to-learn ones.
- **OSS (One-Sided Selection):** Undersamples the majority class while retaining all minority class examples.
- **Normalization:** Structures data to reduce redundancy and improve data integrity.

### Deep Learning Architectures
- **CNN (Convolutional Neural Network):** Operates along the temporal dimension to capture local trends and short-term dependencies.
- **Time CNN:** Uses convolutions along the time dimension to capture short-term patterns and long-term dependencies.
- **FCN (Fully Convolutional Network):** Analyzes sequences of data points, combining strengths of FCNs and RNNs for temporal information.

## Results

The CNN model with **ADASYN** achieved a higher accuracy of 99.82% compared to the CNN model with **SMOTE**, which achieved 99.30%. Both models showed perfect recall of true exoplanets, but the ADASYN model demonstrated significantly better precision (1.00 vs 0.80).

## Future Work

Future research could focus on:
- Fine-tuning ADASYN parameters.
- Exploring combinations of different data augmentation techniques.
- Enhancing model performance through advanced methods.

## Requirements

- Python 3.8
- Imblearn
- Matplotlib
- SciPy
- Seaborn
- TensorFlow/Keras
- scikit-learn
- NumPy
- pandas

