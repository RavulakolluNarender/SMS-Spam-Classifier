# SMS Spam Classifier

This repository contains a Machine Learning (ML) project called "SMS Spam Classifier," which is designed to classify SMS messages as either spam or not spam (ham). The project is implemented using Python and Streamlit to provide a user-friendly web interface for SMS classification.

## Table of Contents

1. [Introduction](#introduction)
2. [Project Overview](#project-overview)
3. [Dataset](#dataset)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Model Building](#model-building)
8. [Web Deployment](#web-deployment)
9. [Improvement](#improvement)
10. [Contributing](#contributing)
11. [License](#license)

## Introduction

Text message (SMS) spam detection is a classic Natural Language Processing (NLP) problem. The goal is to build a model capable of classifying incoming SMS messages as either spam or ham. This project demonstrates how to preprocess text data, build an NLP-based spam classification model, and deploy it as a web application using Streamlit.

## Project Overview

The project follows the following workflow:

1. Data Cleaning: The dataset is loaded and unnecessary columns are dropped. Missing values and duplicate records are handled.

2. Exploratory Data Analysis (EDA): Basic statistics and visualizations are used to understand the distribution of spam and ham messages.

3. Text Preprocessing: The text data is preprocessed, including lowercasing, tokenization, removing special characters, stop words, and applying stemming.

4. Model Building: The preprocessed text data is vectorized using TF-IDF (Term Frequency-Inverse Document Frequency) and then used to train various classification models.

5. Evaluation: The trained models are evaluated based on accuracy and precision scores.

6. Improvement: Different classifiers are compared to select the best-performing model.

7. Web Deployment: The best model is deployed as a web application using Streamlit, allowing users to input SMS messages and receive predictions.

## Dataset

The project uses a dataset containing labeled SMS messages, where each message is categorized as spam or ham. The dataset is preprocessed during the EDA phase to handle missing values and duplicates. Additionally, features like the number of characters, words, and sentences are derived from the text.

## Requirements

The project requires the following libraries and tools:

- Python (>= 3.6)
- Jupyter Notebook (to run the provided code)
- Streamlit (for web deployment)
- NumPy, Pandas, scikit-learn, Matplotlib, Seaborn (for data preprocessing and model building)
- NLTK (Natural Language Toolkit) for text processing
- XGBoost (for one of the classifiers)

## Installation

To run the code and deploy the web application, you need to install the required libraries. Use the following command to install the necessary packages using `pip`:

```bash
pip install streamlit numpy pandas scikit-learn matplotlib seaborn nltk xgboost
```

## Usage

To use the project, follow these steps:

1. Clone this repository to your local machine.
2. Run the provided Jupyter Notebook to explore and preprocess the dataset, build the models, and save the best-performing model.
3. Run the Streamlit application using the provided `app.py` file to deploy the web interface locally.

## Model Building

The project explores several classification algorithms, including:

- Gaussian Naive Bayes
- Multinomial Naive Bayes
- Bernoulli Naive Bayes
- Support Vector Classifier (SVC)
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- AdaBoost Classifier
- Gradient Boosting Classifier
- XGBoost Classifier

Each classifier is trained and evaluated using cross-validation. The best-performing model is selected based on the accuracy score and further deployed in the web application.

## Web Deployment

The best model is deployed as a web application using Streamlit, a Python library for creating interactive web applications. The `app.py` file contains the code for the web interface. To deploy the web application locally, run the following command:

```bash
streamlit run app.py
```

This will start the web application, and you can access it in your browser by navigating to `http://localhost:8501`.

In the web application, you can enter an SMS message and click the "Classify" button to see the prediction whether the message is spam or ham.

## Improvement

The project focuses on building a baseline spam SMS detection model using traditional machine learning algorithms. However, there is room for improvement, including:

1. **Deep Learning Models**: Explore the use of deep learning models, such as recurrent neural networks (RNN) or transformer-based models (e.g., BERT), for improved performance.

2. **Feature Engineering**: Experiment with additional text features or handcrafted features that could improve model performance.

3. **Ensemble Methods**: Implement ensemble methods, such as stacking or voting classifiers, to combine the predictions of multiple models for better accuracy.

4. **Hyperparameter Tuning**: Optimize the hyperparameters of the selected model using techniques like grid search or random search to improve its performance.

## Contributing

If you would like to contribute to the project, you can follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make the necessary changes and commit your code.
4. Push the changes to your forked repository.
5. Submit a pull request describing your changes.

## License

The project is released under the [MIT License](LICENSE). Feel free to use, modify, and distribute the code for your own purposes.
