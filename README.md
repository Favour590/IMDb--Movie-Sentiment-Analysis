IMDb Movie Review Sentiment Analysis

Project Overview

This project uses Natural Language Processing (NLP) and machine learning techniques in R to classify IMDb movie reviews as either positive or negative.

The dataset contains 50,000 movie reviews, with an equal distribution of positive and negative sentiments.

Project Objectives

* Explore and understand the IMDb movie review dataset.
* Clean and preprocess textual data.
* Perform sentiment analysis and text exploration.
* Apply TF-IDF for feature extraction.
* Build and evaluate machine learning classification models.
* Interpret model predictions using LIME and SHAP.

Data Preparation

The dataset was cleaned by removing duplicate reviews and preprocessing the text. After removing 418 duplicate reviews, 49,582 reviews remained for analysis.

Text preprocessing included:

* Converting text to lowercase.
* Removing punctuation, numbers and HTML tags.
* Removing stop words.
* Tokenizing reviews into individual words.

Machine Learning Models

Four classification models were developed and evaluated:

1. Logistic Regression
2. Multinomial Naive Bayes
3. Support Vector Machine (SVM)
4. Ridge-Regularized Logistic Regression

Model Performance

Model	Accuracy	ROC-AUC
Logistic Regression	85.12%	0.8594
Naive Bayes	85.67%	0.9232
SVM	86.42%	0.9339
Ridge-Regularized Logistic Regression	88.25%	0.9477

Ridge-Regularized Logistic Regression achieved the highest accuracy and ROC-AUC among the four models.

Model Interpretation

LIME and coefficient-based SHAP analysis were used to examine how words contributed to the model’s predictions.

Tools and Technologies

* R and RStudio
* tidyverse
* tidytext
* text2vec
* glmnet
* LiblineaR
* pROC
* lime

Project Files

* IMDb_Sentiment_Analysis.R: R script containing the analysis and modelling code.
* IMDb_Sentiment_Analysis.Rmd: R Markdown document containing code, explanations and outputs.
* IMDb_Sentiment_Analysis.html: Knitted HTML report.

Conclusion

The project demonstrates how text preprocessing, sentiment analysis and machine learning can be used to classify movie reviews. Among the models evaluated, Ridge-Regularized Logistic Regression produced the strongest overall performance.
