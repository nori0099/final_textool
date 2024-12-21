#!/bin/bash

INPUT_FILE="kdrama_list.csv"
OUTPUT_FILE="extracted_name_score.csv"

# Extract specified columns using csvcut
csvcut -c "Name,Score" "$INPUT_FILE" > "$OUTPUT_FILE"

echo "Extracted columns saved to $OUTPUT_FILE"

