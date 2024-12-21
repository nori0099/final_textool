#!/usr/bin/env python3

import pandas as pd
import numpy as np

# Get feature importance from the Random Forest model
feature_importances = model.feature_importances_

# Map feature indices to words using the TF-IDF vectorizer
feature_names = vectorizer.get_feature_names_out()

# Combine feature names and their importance into a DataFrame
importance_df = pd.DataFrame({
    'word': feature_names,
    'importance': feature_importances
})

# Sort by importance in descending order
importance_df = importance_df.sort_values(by='importance', ascending=False)

# Display the top 20 most important features
print(importance_df.head(20))

