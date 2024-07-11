import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
import joblib

# Load the data
try:
    data = pd.read_excel('C:\\Users\\00078411\\Downloads\\Copy of Salvage Data.xlsx')  # Replace with your actual file path
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Ensure the dataset is not empty
if data.empty:
    print("The dataset is empty. Please check the data source.")
    exit()

# Replace '-' in 'Nos' with 1 and convert to numeric, handling any non-numeric values
data['Nos'] = data['Nos'].replace('-', 1).fillna(1).astype(float)

# Convert 'Lot Creation dt' and 'Dt of sale' to datetime
data['Lot Creation dt'] = pd.to_datetime(data['Lot Creation dt'])

# Fill missing values in 'Item Name' with 'Unknown'
data['Item Name'] = data['Item Name'].fillna('Unknown')

# Extract year, month, and day from 'Lot Creation dt'
data['Year'] = data['Lot Creation dt'].dt.year
data['Month'] = data['Lot Creation dt'].dt.month
data['Day'] = data['Lot Creation dt'].dt.day

# Extract day of the week
data['DayOfWeek'] = data['Lot Creation dt'].dt.dayofweek

# Apply TF-IDF to the 'Item Name' column
tfidf = TfidfVectorizer(max_features=100)  # Adjust max_features as needed
item_name_tfidf = tfidf.fit_transform(data['Item Name'].fillna('')).toarray()

# Create a DataFrame from the TF-IDF features and concatenate with the original data
item_name_tfidf_df = pd.DataFrame(item_name_tfidf, columns=tfidf.get_feature_names_out())
data = pd.concat([data.reset_index(drop=True), item_name_tfidf_df], axis=1)

# Prepare the features and target variable
features = ['Nos', 'Wt', 'Lot rejection count', 'Year', 'Month', 'Day', 'DayOfWeek'] + list(item_name_tfidf_df.columns)
X = data[features]
y = data['Material Value']

# Handle NaN values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model, scaler, tfidf vectorizer, and imputer to disk
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(tfidf, 'tfidf.pkl')
joblib.dump(imputer, 'imputer.pkl')

# Print the feature importances
feature_importances = model.feature_importances_
print("Feature Importances:")
for feature, importance in zip(features, feature_importances):
    print(f"{feature}: {importance:.4f}")