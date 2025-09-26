# 🌍 Air Quality & Health Risk Predictor

## 📖 Project Description
This project uses **Machine Learning** to predict **Air Quality Index (AQI) categories** and evaluate related **health risks** based on pollutant and meteorological data collected from Indian cities.  
A **Random Forest Classifier** was trained on historical air quality data and achieved near-perfect accuracy in classifying AQI into categories such as *Good, Fair, Moderate, Poor, and Very Poor*.  

The project pipeline includes **data preprocessing, exploratory data analysis (EDA), model training, evaluation, and deployment-ready model saving (`.pkl`)**.

---

## 🔍 Project Overview
1. **Dataset**  
   - Source: India Air Quality Dataset  
   - Features: Pollutants (PM2.5, PM10, NO2, SO2, CO, Ozone, etc.) and meteorological factors  
   - Target: **AQI Category** (Good, Fair, Moderate, Poor, Very Poor)  

2. **Exploratory Data Analysis (EDA)**  
   - Distribution of pollutants and AQI categories  
   - Correlation analysis of pollutants with AQI  
   - AQI category trends across regions  

3. **Model Building**  
   - Algorithm: **Random Forest Classifier**  
   - Data Split: 70% Training, 30% Testing  
   - Performance:  
     - ✅ Train Accuracy: **1.0**  
     - ✅ Test Accuracy: **99.97%**  
     - ✅ Perfect classification across AQI categories  

4. **Model Evaluation**  
   - Confusion Matrix Heatmap → Almost zero misclassification  
   - Classification Report → Precision, Recall, F1-score = **1.0** for all classes  
   - Feature Importance → PM2.5, PM10, and NO2 emerged as key contributors  

5. **Deployment-Ready Model**  
   - Trained model saved as **`air_quality_model.pkl`**  
   - Can be used in a **Streamlit App**, **Flask API**, or dashboards for real-time AQI prediction  

---

## 📊 Key Insights
- **PM2.5, PM10, and NO2** are the dominant pollutants impacting AQI levels.  
- Machine Learning can effectively classify AQI and support **public health awareness**.  
- The approach demonstrates how **data-driven insights** can help in **environmental monitoring and policy-making**.  

---

## 📂 Repository Structure
