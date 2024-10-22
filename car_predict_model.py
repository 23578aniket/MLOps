import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import pickle

# Load the dataset
cars_df = pd.read_csv("cars24-car-price.csv")

# Feature selection - including the full name of the car
features = ['full_name', 'year', 'km_driven', 'fuel_type', 'engine', 'seller_type', 'transmission_type', 'mileage',
            'max_power', 'seats']
X = cars_df[features]
y = cars_df['selling_price']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing for categorical and numerical features
categorical_features = ['full_name', 'fuel_type', 'seller_type', 'transmission_type']
numerical_features = ['year', 'km_driven', 'engine', 'mileage', 'max_power', 'seats']

# Create a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),  # Standard scaling for numerical features
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # One-hot encoding for categorical features, including full car name
    ])

# Create a pipeline that first transforms the data, then fits the model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge(positive=True))  # Ensures positive predictions
])

# Train the model
pipeline.fit(X_train, y_train)

# Save the trained model to a pickle file
model_filename = 'car_price_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(pipeline, file)

print(f"Model trained and saved successfully as {model_filename}!")


# Function to predict car price
def predict_price(input_data):
    # Load the trained model
    with open(model_filename, 'rb') as file:
        loaded_model = pickle.load(file)

    # Ensure input_data is a pandas DataFrame with the correct columns
    input_df = pd.DataFrame(input_data)

    # Pass the input DataFrame to the model for prediction
    return loaded_model.predict(input_df)


# Example usage: Making predictions based on user input
example_input = {
    'full_name': ['Maruti Alto Std'],  # Full car name, not just the brand
    'year': [2012],
    'km_driven': [120000],
    'fuel_type': ['Petrol'],
    'engine': [796],
    'seller_type': ['Individual'],
    'transmission_type': ['Manual'],
    'mileage': [19.7],
    'max_power': [46.3],
    'seats': [5]
}

# Convert example input to DataFrame and make prediction
example_input_df = pd.DataFrame(example_input)

# Make predictions
predicted_price = predict_price(example_input_df)

