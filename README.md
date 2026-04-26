## Spam Detection with Explainable AI (LIME)

This project builds a machine learning model to classify SMS messages as spam or not spam (ham).  
It also uses LIME (Local Interpretable Model-agnostic Explanations) to explain why the model makes certain predictions.

---

## Objectives
- Build a text classification model for spam detection
- Evaluate model performance
- Explain model predictions using LIME
- Visualize feature importance (word-level explanation)

---

## Tools & Technologies
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- LIME
- Matplotlib

---

## Dataset
SMS Spam Collection Dataset  
- Labels: spam / ham  
- Text messages collected from real SMS data

---

## Workflow

1. Load dataset
2. Data preprocessing
3. Convert text using TF-IDF
4. Train Logistic Regression model
5. Evaluate model performance
6. Apply LIME for explainability
7. Visualize explanations using bar chart

---

## Model Performance
- Logistic Regression used for classification
- Evaluation metrics: Accuracy, Precision, Recall, F1-score

---

## Explainability (LIME)

LIME is used to explain individual predictions by showing which words contributed most to the model’s decision.

Example:
- Words like "free", "win", "prize" → strong indicators of spam
- Words like "friend", "hello" → more likely ham

---

## Visualization

Feature importance is displayed using bar charts:
- Positive values → support prediction
- Negative values → oppose prediction

---

## 🚀 Key Insight

This project demonstrates not only how to build a spam classifier, but also how to interpret AI decisions, making the model more transparent and trustworthy.
