# Amazon Product Review Sentiment Analysis
<img width="2533" height="1280" alt="Screenshot 2026-01-19 091617" src="https://github.com/user-attachments/assets/db9ece2f-73ae-4567-8f4e-0fad401375b7" />

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-NLP-green)
![Status](https://img.shields.io/badge/Project-Active-success)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-orange)
![Deployment](https://img.shields.io/badge/Deployment-Streamlit%20Cloud-red)

---
# ACCESS THE APP HERE : https://sentiment-analysis-machine-learning-model-r3axhcsupyyk4opeazm6.streamlit.app/
## 📌 1. Problem Definition

### What is Sentiment Analysis?

Sentiment Analysis is a Natural Language Processing (NLP) task that automatically determines the emotional tone expressed in textual data. In e-commerce platforms like Amazon, it is used to classify customer reviews into **Positive**, **Neutral**, or **Negative** sentiments based on the opinion conveyed in the text.

Customer feedback is subjective by nature and often contains informal language, ambiguity, and mixed opinions. Manually reviewing hundreds of thousands of reviews is not scalable. Sentiment analysis enables automated extraction of meaningful insights from large volumes of unstructured text data.

---
![Uploading Screenshot 2026-01-06 000041.png…]()

### Business Understanding

Amazon receives millions of product reviews that directly impact:

* Customer purchase decisions
* Product rankings and recommendations
* Seller performance metrics
* Brand trust and customer satisfaction

By applying sentiment analysis, Amazon can:

* Detect product or service issues early
* Identify recurring customer pain points
* Understand what features customers value most
* Enable data-driven decisions across product, marketing, and operations teams

Transforming raw customer feedback into structured sentiment signals allows the business to move from reactive feedback handling to proactive product and service improvement.

---

### Problem Statement

Build a machine learning system that automatically classifies an Amazon product review into one of three sentiment categories:

* **Negative**
* **Neutral**
* **Positive**

The system should be scalable, reusable, and deployable for real-time inference.

---

### Objective

**Primary Objective**

* Predict the sentiment of a given Amazon product review accurately.

**Secondary Objectives**

* Perform exploratory data analysis (EDA) to understand sentiment distribution
* Extract insights explaining customer satisfaction or dissatisfaction
* Build an end-to-end NLP pipeline from raw text to deployment
* Persist trained models for reuse in production

---

### Machine Learning Task

* **Learning Type:** Supervised Learning
* **Task:** Multi-Class Text Classification
* **Input:** Customer review text
* **Output:** Sentiment label (Negative / Neutral / Positive)

---

### Evaluation Metrics

* Accuracy
* Confusion Matrix
* (Extended evaluation: Precision, Recall, F1-Score per class)

---

## 📊 2. Data Collection

### Dataset: Amazon Fine Food Reviews

**Source:** Kaggle – Amazon Fine Food Reviews Dataset

This dataset contains customer reviews of fine food products sold on Amazon over a span of more than 10 years.

---

### Dataset Overview

* **Time Period:** October 1999 – October 2012
* **Total Reviews:** 568,454
* **Unique Users:** 256,059
* **Unique Products:** 74,258

The dataset includes reviews across multiple Amazon categories and contains both structured and unstructured data.

---

### Dataset Attributes

* **ProductId** – Unique identifier for the product
* **UserId** – Unique identifier for the reviewer
* **ProfileName** – Name of the reviewer
* **HelpfulnessNumerator** – Number of users who found the review helpful
* **HelpfulnessDenominator** – Number of users who voted on the review
* **Score** – Rating given by the user (1 to 5)
* **Time** – Unix timestamp of the review
* **Summary** – Short summary of the review
* **Text** – Full review text

---

## 🏗️ Project Workflow (High-Level)

1. Problem definition & business understanding
2. Data ingestion and cleaning
3. Exploratory Data Analysis (EDA)
4. Text preprocessing and normalization
5. Feature extraction (TF-IDF / NLP features)
6. Model training and evaluation
7. Model persistence (.pkl)
8. Inference pipeline creation
9. Streamlit UI deployment
10. Insight generation and documentation

---
Executive EDA Summary (One Paragraph)

Exploratory analysis reveals a large, text-rich, but sentiment-imbalanced dataset dominated by positive reviews. Review length varies systematically with sentiment, with negative and neutral reviews providing more detailed feedback. The vocabulary size and text volume support classical NLP pipelines (TF-IDF + linear models) as a strong and interpretable baseline, while deep learning models can be explored as extensions.

---
| Rating | Sentiment | Rationale                   |
| ------ | --------- | --------------------------- |
| 1–2    | Negative  | Clear dissatisfaction       |
| 3      | Neutral   | Ambiguous / mixed sentiment |
| 4–5    | Positive  | Clear satisfaction          |

---
Text preprocessing was implemented using a modular pipeline including lowercasing, URL and punctuation removal, stopword filtering, and stemming. All transformations were applied post–label freeze to prevent leakage and ensure reproducibility during inference.
<img width="1649" height="895" alt="Screenshot 2026-01-05 135442" src="https://github.com/user-attachments/assets/2df5a3ba-5a03-400c-9306-70d99de79701" />

---
### STEP 8 — Feature Extraction & Baseline Modeling


Perform a stratified train/validation split

Vectorize text using TF-IDF

Train a class-weighted Logistic Regression

Establish a baseline benchmark (accuracy + per-class metrics)

## Class-wise Performance (Critical)
### Positive
Precision: 0.97

Recall: 0.85

F1: 0.91

The model is extremely reliable when predicting positive sentiment, which is expected given class dominance and strong lexical signals.

### Negative

Precision: 0.69

Recall: 0.76

F1: 0.72

Negative sentiment is captured reasonably well. Recall is higher than precision, meaning the model prefers to flag potential negatives rather than miss them—acceptable from a business-risk perspective.

### Neutral (Hard Class)

Precision: 0.29

Recall: 0.60

F1: 0.39

Neutral sentiment is the most challenging class. Many neutral reviews are confused with positive or negative, reflecting the inherently ambiguous nature of 3-star reviews.


A TF-IDF + class-weighted Logistic Regression baseline achieved ~82% accuracy with strong performance on positive and negative sentiment. Neutral sentiment remains the most difficult class due to semantic ambiguity. This baseline provides a strong, interpretable benchmark suitable for production comparison.



How to Build and Run (Docker)
Build image
docker build -t amazon-sentiment-api .

Run container
docker run -p 5000:5000 amazon-sentiment-api


Test:
curl http://localhost:5000/health
