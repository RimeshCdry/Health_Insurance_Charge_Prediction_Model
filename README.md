# Health Insurance Charge Prediction using Linear Regression 💰📊  

## Overview 📄  

This project predicts **health insurance charges** using a **Linear Regression model** based on factors like **age, BMI, smoking status, and region**. The workflow includes **data collection, preprocessing, feature scaling, model training, and evaluation** using metrics like **R² score, Mean Squared Error (MSE), and Mean Absolute Error (MAE)**. Additionally, a **Streamlit web application** has been developed for real-time predictions.  

---

## Project Workflow 🔄  

### 1. Data Collection 📂  
- The dataset was obtained from **Kaggle**.  
- Features include:  
  - **Age** (Years)  
  - **BMI** (Body Mass Index)  
  - **Smoking Status** (Smoker/Non-Smoker)  
  - **Region** (Geographical location)  
  - **Insurance Charges** (Target variable)  

### 2. Data Preprocessing 🔍  
- Handled missing values (if any).  
- Converted categorical variables (e.g., **smoking status, region**) into numerical format using **Label Encoder**.  

### 3. Feature Scaling 📊  
- Applied **StandardScaler** to normalize numerical features like **age** and **BMI** for better model performance.  

### 4. Train-Test Split 🔢  
- The dataset was split into **training (80%)** and **testing (20%)** sets.  

### 5. Model Training 🤖  
- Trained a **Linear Regression model** using **Scikit-Learn**.  
- The model was optimized to predict insurance charges based on input features.  

### 6. Model Evaluation 📉  
- Evaluated model performance using:  
  - **R² Score** (Coefficient of Determination)  
  - **Mean Squared Error (MSE)**  
  - **Mean Absolute Error (MAE)**  

### 7. Real-Time Prediction 🔮  
- Developed a **Streamlit web app** where users can input their details and get an insurance charge estimate.  

---

## UI Development 🖥️  
- A **Graphical User Interface (GUI)** was built using **Streamlit**.  
- Users can enter their **age, BMI, smoking status, and region** to get a real-time **insurance cost prediction**.  

---

## Technologies Used 💻  
- **Python** (Pandas, NumPy, Scikit-Learn, joblib)  
- **Streamlit** (for UI)  
- **Plotly** (for visualization)  
- **Jupyter Notebook** (for model training & evaluation)  

---

## Installation Guide 🛠️  

1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/Health_Insurance_Charge_Prediction.git
   cd Health_Insurance_Charge_Prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit UI:
   ```bash
   streamlit run app.py
   ```
---

## User Guide🚀:
  - Open the Streamlit UI in a browser.
  - Enter details such as age, BMI, smoking status, and region.
  - Click on Predict to get estimated insurance charges.
  - View Result.

---

## Results & Insights 📊:
  - The **Linear Regression model** provides reliable insurance charge predictions.
  - The web app allows users to check their estimated costs easily.
  - This model can assist in **financial planning** and **insurance cost analysis**.

---

## Future Enhancements 🚧
 - Add more features like medical history to improve accuracy.
 - Deploy the model on a cloud platform (e.g., AWS, Heroku).
 - Experiment with advanced ML models (e.g., XGBoost, Random Forest).

---

## 👥Contact
For questions or feedback, feel free to reach out:
  - GitHub: @RimeshCdry
  - Email: rimeshcdry45@gmail.com
  - LinkedIn: https://www.linkedin.com/in/rimesh-chaudhary-09a25a30a
