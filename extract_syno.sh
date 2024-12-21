#!/bin/bash

INPUT_FILE="kdrama_list.csv"
OUTPUT_FILE="cleaned_syno.csv"

# Extract specified columns using csvcut
csvcut -c "Sinopsis" "$INPUT_FILE" | sed 's/"//g; s/\r//g; s/—/ /g; s/\\\\/'"'"'/g; s/[^a-zA-Z0-9 ]//g' | tr '[:upper:]' '[:lower:]'| tr -d ',' > "$OUTPUT_FILE"

echo "Extracted columns saved to $OUTPUT_FILE"
