# Refurbished Car Price Predictor

A Streamlit web application that predicts the estimated price of a refurbished car based on user inputs such as car model, year, kilometers driven, fuel type, seller type, and transmission. The app uses a machine learning model (Random Forest Regressor) trained on a dataset of refurbished cars.

---

## Live Demo

Try the app live here: [Refurbished Car Price Predictor](https://refurbished-car-price-predictor-nikhil-reddy.streamlit.app/)

---

## Features

- **User-friendly input form:** Enter car details through an intuitive sidebar interface.
- **Price Prediction:** Uses a trained Random Forest Regression model to estimate the selling price of a refurbished car.
- **Dynamic Progress Bar:** Displays a loading animation while predicting.
- **Search Link:** Provides a direct Google search link to explore more about the car model.
- **Similar Cars:** Displays similar cars from the dataset within the predicted price range for comparison.
- **Expandable Information:** Sections explaining the app features and usage instructions.

---

## Dataset
The model is trained on a dataset (car_data.csv) containing features such as:
- **Car_Name**
- **Year**
- **Kms_Driven**
- **Fuel_Type**
- **Seller_Type**
- **Transmission**
- **Selling_Price (target variable)**
The dataset is cleaned by dropping irrelevant columns like Owner and Present_Price.

## Model
### Algorithm: Random Forest Regressor
- **Preprocessing:** One-hot encoding of categorical variables (Car_Name, Fuel_Type, Seller_Type, Transmission), numerical features passed through as-is.
- **Training/Test Split:** 80% training, 20% testing
- **Implementation:** Combined in a scikit-learn Pipeline for preprocessing and regression.
### Technologies Used
- **Python 3.12**
- **Streamlit for the web interface**
- **Pandas for data handling**
- **scikit-learn for machine learning and preprocessing**
- **joblib (optional) for model persistence**
