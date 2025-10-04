# Project 1: Robust Feedback Classification (COMP9417)

This project was completed as part of the **COMP9417 (Machine Learning)** course at UNSW.  
It was a group project where we aimed to build a robust classification model to automatically classify customer feedback into 28 product-related departments, under the challenges of **class imbalance** and **distribution shift**.

---

## Goal
- Automatically classify feedback into product categories.
- Overcome issues of **severe class imbalance** (some classes with thousands of samples, some with <10).
- Mitigate the impact of **distribution shift** between training and test sets.

---

## Methods
- **Data Preprocessing**  
  - Feature importance ranking using Random Forest.  
  - Feature selection: top 200 features chosen for model input.  
  - Standardization of features to improve model stability.  

- **Models Tested**  
  - Logistic Regression  
  - Support Vector Machine (SVM)  
  - Deep Neural Network (DNN)  

- **Improvement Techniques**  
  - Oversampling (Random Oversampling, SMOTE)  
  - Focal Loss to improve minority class recognition  
  - Optimized DNN architecture with selected features  

---

## Results
- **Improved Logistic Regression** → Accuracy 0.77, Unweighted F1 0.4657  
- **Oversampled SVM** → Accuracy 0.77, Unweighted F1 0.4618  
- **Low Bias DNN (with Focal Loss)** → Accuracy 0.78, Unweighted F1 0.4726, Cross-Entropy Loss 0.0068  

✅ The **Low Bias DNN with Focal Loss** achieved the best balance between majority and minority classes.

---

## My Contribution
I was responsible for the **RF_DNN (Random Forest + Deep Neural Network) pipeline**:
- Used Random Forest to calculate feature importance and selected the top 200 features.  
- Preprocessed features with `StandardScaler` for normalized inputs.  
- Designed and trained a Deep Neural Network with optimized architecture.  
- Integrated **Focal Loss** to address class imbalance, improving minority class recall.  
- Delivered the best overall performance in the group:  
  - Validation Accuracy: **0.78**  
  - Unweighted F1: **0.4726**  
  - Cross-Entropy Loss: **0.0068**  

---

## Tools & Skills
- Python, Scikit-learn  
- Random Forest, Logistic Regression, SVM, DNN  
- Techniques: Feature Selection, Oversampling, Focal Loss  
- Evaluation Metrics: Accuracy, Macro/Unweighted F1, Cross-Entropy Loss  

---

## Report
Full project report: [`9417_report.pdf`](../9417%20report.pdf)  
