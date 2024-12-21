#!/usr/bin/env python3

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

df = pd.read_csv("combine_complete.csv")

# Check for missing value
print(f"Missing values in processed_text: {df['processed_text'].isna().sum()}")

# Fill missing values with an empty string
df['processed_text'] = df['processed_text'].fillna("")


vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['processed_text']).toarray()
y = df['Score']

# data split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_val, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

# Get feature importance
feature_importances = model.feature_importances_

# Get the feature words
feature_names = vectorizer.get_feature_names_out()

# Combine feature names and their importance scores
importance_df = pd.DataFrame({
    'word': feature_names,
    'importance': feature_importances
})

# Sort 
importance_df = importance_df.sort_values(by='importance', ascending=False)

#the top 10 most important features
print(importance_df.head(10))

