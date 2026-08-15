# Real-Time Credit Card Fraud Detection with Synthetic Data Simulation

This project implements an end-to-end Machine Learning pipeline to detect fraudulent credit card transactions using the highly imbalanced **IEEE-CIS Fraud Detection dataset**. Developed as part of the **AI4ALL Ignite Accelerator**, the project addresses real-world constraints such as heavy feature anonymization and severe class imbalance. 

To showcase model performance interactively, the project features a **Streamlit Web Application** equipped with a synthetic transaction generator, allowing users to tweak transaction conditions in real time and receive instant risk probabilities.

---

## Problem Statement Financial fraud causes billions of dollars in losses annually, but detecting it is challenging because fraudulent transactions account for less than **3.5% of total transactions**. Furthermore, financial datasets heavily anonymize features ($V1-V339$, $C$-counts, $D$-deltas) to protect consumer privacy. 

The goal of this project is to build an accurate classification pipeline capable of identifying high-risk transactions while minimizing false positives, and to deploy an interactive interface that allows users to evaluate risk despite anonymized input limitations.

---

## Key Results 1. **Engineered 17+ domain-specific features**, including 24-hour transaction velocity, card account age, late-night transaction flags (2 AM–5 AM), and purchaser/recipient email domain matching.
2. **Evaluated Multiple Models**: Trained and benchmarked XGBoost and Neural Network architectures on heavily preprocessed anonymized features.
3. **High Performance**: Achieved strong fraud detection performance evaluated via **ROC-AUC** and **F1-Score**, with **V258**, **V70**, and **V294** identified as the top predictive anonymized signals.
4. **Interactive Deployment**: Developed and deployed a Streamlit dashboard featuring single, interactive, and batch synthetic data generators to enable dynamic inference testing.

---

## Methodologies * **Data Cleaning & Imputation**: Handled missing values across 300+ features using median and mode imputation strategies.
* **Feature Engineering**: Formulated velocity metrics (`Transaction_velocity_24hr`), card age indicators (`Card_age_days`), frequency encoding (`card1_freq`, `addr1_freq`), and risk indicators (`is_late_night`, `email_domain_match`).
* **Model Training & Optimization**: Trained XGBoost and Keras Neural Networks on preprocessed data.
* **Deployment & Synthetic Simulation**: Built a Streamlit web application that synthesizes anonymized $V$-columns according to dataset distributions, allowing real-time prediction scores (`model.predict_proba`).

---

## Data Sources * **IEEE-CIS Fraud Detection Dataset**: Hosted on [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection), containing 590,540 real-world transaction records.

---

## Technologies Used * **Python**
* **Pandas & NumPy** (Data Manipulation & Engineering)
* **Scikit-Learn & XGBoost** (Model Building & Feature Importance)
* **TensorFlow / Keras** (Deep Learning Architecture)
* **Streamlit** (Interactive Application Deployment)
* **Seaborn & Matplotlib** (Visualization)

---

## Application Demo & Artifacts

* 🚀 **Live Interactive App**: [Click Here to Open Streamlit Demo]([https://your-streamlit-app-link.streamlit.app](https://ai4all-2a-fraud-detection.streamlit.app/)) 

![Streamlit App Screenshot](docs/assets/streamlit_demo.gif)
*Caption: Real-time fraud probability scoring via synthetic feature generation.*

---

## Authors This project was completed in collaboration with:
* **Jolaoluwa Amodu** ([LinkedIn](https://www.linkedin.com/in/jola-amodu/) | [GitHub Profile](https://github.com/the-jola-amodu))
* **Annika Bhatia** ([LinkedIn](https://www.linkedin.com/in/annika-bhatia/) | [GitHub Profile](https://github.com/annikabhatia))
* **Patrick Selby** ([LinkedIn](https://www.linkedin.com/in/patrick-ennin-selby-136253301/) | [GitHub Profile](https://github.com/pat-selby))
