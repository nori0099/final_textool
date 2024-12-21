#!/usr/bin/env python3

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')

# Load the cleaned text from the CSV file
df = pd.read_csv("cleaned_syno.csv")
lines = df["sinopsis"].tolist()

# Tokenize and remove stopwords
stop_words = set(stopwords.words('english'))
processed_lines = []

for line in lines:
    tokens = word_tokenize(line)
    filtered_tokens = [word for word in tokens if word not in stop_words]  
    processed_lines.append(" ".join(filtered_tokens))

# Save the processed lines to a new CSV file
processed_df = pd.DataFrame(processed_lines, columns=["processed_text"])
processed_df.to_csv("processed_synopses.csv", index=False)
