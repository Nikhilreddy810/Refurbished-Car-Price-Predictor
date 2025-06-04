import streamlit as st
import joblib
import pandas as pd
import urllib.parse
import time
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv('car_data.csv')

# Drop 'Owner' column if it exists
if "Owner" in df.columns:
    df.drop("Owner", axis=1, inplace=True)

if "Present_Price" in df.columns:
    df.drop("Present_Price", axis=1, inplace=True)


# Features and target
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Define categorical and numerical columns
categorical_cols = ['Car_Name', 'Fuel_Type', 'Seller_Type', 'Transmission']
numerical_cols = ['Year', 'Kms_Driven']

# Preprocessing pipeline: one-hot encode categorical, passthrough numerical
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'  # numerical columns are passed as is
)

# Build full pipeline with model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Streamlit UI starts here
st.title("Refurbished Car Price Predictor")

st.sidebar.header("Enter Car Details")
car = st.sidebar.text_input("Car Model", "swift")
year = st.sidebar.number_input("Year", min_value=1900, max_value=2025, value=2018)
kms = st.sidebar.number_input("Kms Driven", min_value=0, value=25000)
fuel = st.sidebar.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG'])
seller = st.sidebar.selectbox("Seller Type", ['Dealer', 'Individual'])
trans = st.sidebar.selectbox("Transmission", ['Manual', 'Automatic'])

input_data = pd.DataFrame({
    'Car_Name': [car],
    'Year': [year],
    'Kms_Driven': [kms],
    'Fuel_Type': [fuel],
    'Seller_Type': [seller],
    'Transmission': [trans],
})

with st.expander("About This WebApp"):
    st.write(
        "This Refurbished Car Price Predictor estimates the price of a refurbished car "
        "based on model, year, kms driven, fuel type, seller type, and transmission."
    )
    st.header("Key Features:")
    st.write("1. User input form")
    st.write("2. Machine Learning model (RandomForestRegressor)")
    st.write("3. Model trained on a dataset of refurbished cars")
    st.write("4. Predicts car price based on user inputs")
    st.write("5. Provides a link to explore more about the car model")

with st.expander("How to Use"):
    st.write("1. Enter the car model name, year, kms driven, fuel type, seller type, and transmission type in the sidebar.")
    st.write("2. Click on 'Predict Price' to get the estimated price of the refurbished car.")
    st.write("3. Click on the link provided to explore more about the car model.")

st.write("---")

if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]

    # Progress bar animation
    latest_iteration = st.empty()
    bar = st.progress(0)
    for i in range(100):
        bar.progress(i + 1)
        time.sleep(0.01)
    st.success(f"Estimated Car Price: ₹{prediction:.2f} Lakhs")

    # Google search link
    if car.strip():
        query = urllib.parse.quote(car)
        google_url = f"https://www.google.com/search?q={query}"
        st.markdown(f"[🔎 Click here to explore more about {car}]({google_url})", unsafe_allow_html=True)

    st.write("---")

    # Show similar cars within price range
    price_range = [max(0, prediction - 2), prediction + 2]  # avoid negative price
    similar_data = df[(df["Selling_Price"] >= price_range[0]) & (df["Selling_Price"] <= price_range[1])]
    if not similar_data.empty:
        st.write("### Similar cars with predicted price range:")
        st.dataframe(similar_data.sample(min(5, len(similar_data))))
else:
    st.info("Fill in the details and click 'Predict Price' to see the estimated car price.")
