import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import os

# Get the absolute path to the bias-checker folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'sample_data.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# 1. Load Data
if not os.path.exists(DATA_PATH):
    print(f"❌ ERROR: File not found at {DATA_PATH}. Please create the CSV first.")
else:
    df = pd.read_csv(DATA_PATH)

    # 2. Encode Gender
    encoder = LabelEncoder()
    encoded_gender = encoder.fit_transform(df['gender'])

    # 3. Setup Scaler with FIXED column names
    # This ensures the scaler expects 'gender', not 'gender_encoded'
    X = pd.DataFrame({
        'gender': encoded_gender,
        'age': df['age'],
        'income': df['income']
    })

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y = df['approved']

    # 4. Train Model
    model = LogisticRegression()
    model.fit(X_scaled, y)

    # 5. Save assets directly into the model folder
    pickle.dump(model, open(os.path.join(MODEL_DIR, 'model.pkl'), 'wb'))
    pickle.dump(encoder, open(os.path.join(MODEL_DIR, 'encoder.pkl'), 'wb'))
    pickle.dump(scaler, open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb'))

    print("✅ Success: Model trained and saved. Column names set to: gender, age, income.")