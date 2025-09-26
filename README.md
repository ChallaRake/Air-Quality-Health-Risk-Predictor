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

## 🛠️ Technologies Used
- **Python** 🐍
- **Pandas, NumPy, Matplotlib, Seaborn** (EDA & Visualization)
- **Scikit-learn** (Machine Learning Model)
- **Joblib** (Model Serialization)
- **Jupyter Notebook**


##📌 Future Enhancements
- Build a **Streamlit Web App** for interactive **AQI** prediction
- Expand dataset with **real-time** air quality **APIs**
- Integrate **health risk level warnings** for sensitive groups

---

## 📂 Repository Structure

Air_Quality_Health_Risk_Predictor/\
│── india_air_quality_data.csv.gz/ # Dataset (CSV files)\
│── notebooks/ # Jupyter notebooks (EDA & Model building)\
|   └── AQI API scrape.ipynb
|   └── ML AIR QUALITY.ipynb
│── air_quality_model.zip/ # Saved model (air_quality_model.pkl)\
│── Air-Quality-Health-Risk-Predictor_PPT_Main/ (PPT File)\
│── app.py/ # Streamlit/Flask app (optional)\
│── README.md # Project description & usage\
│── requirements.txt # Dependencies\

<img width="1919" height="1096" alt="Screenshot 2025-09-26 100101" src="https://github.com/user-attachments/assets/962b7175-b597-40af-b2a0-9dbcfe3ce0ca" />

<img width="1919" height="1096" alt="Screenshot 2025-09-26 100119" src="https://github.com/user-attachments/assets/47acf48c-def7-4360-b581-0a7b5baa1c99" />

<img width="1919" height="1095" alt="Screenshot 2025-09-26 100138" src="https://github.com/user-attachments/assets/586531c7-1d26-4c79-95ea-3473946d862e" />

<img width="1919" height="1098" alt="Screenshot 2025-09-26 100202" src="https://github.com/user-attachments/assets/fdbeed7c-2a9e-497f-939e-0472dfd5bea0" />

## Scan this below QR code to explore the live AQI Predictor Pro web app.

<img width="1318" height="1714" alt="Untitled (2)" src="https://github.com/user-attachments/assets/9bfac782-4c3b-4e04-a0be-199fa14f1cc0" />


