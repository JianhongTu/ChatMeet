#!/bin/bash

# Script to generate Environment.swift from .env file
# This runs as a build phase script in Xcode

ENV_FILE="${SRCROOT}/.env"
OUTPUT_FILE="${SRCROOT}/ChatMeet/Environment.swift"

echo "Generating Environment.swift from .env file..."

# Check if .env exists
if [ ! -f "$ENV_FILE" ]; then
    echo "error: .env file not found at ${ENV_FILE}"
    echo "error: Please copy .env.example to .env and add your credentials"
    exit 1
fi

# Start generating the Swift file
cat > "$OUTPUT_FILE" << 'EOF'
//
// Environment.swift
// ChatMeet
//
// Auto-generated from .env file. DO NOT EDIT MANUALLY.
//

import Foundation

enum Environment {
EOF

# Read .env file and generate Swift constants
while IFS='=' read -r key value; do
    # Skip comments and empty lines
    if [[ ! "$key" =~ ^#.* ]] && [[ -n "$key" ]]; then
        # Remove quotes and whitespace from value
        value=$(echo "$value" | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//" -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
        
        # Add to Swift file
        echo "    static let ${key} = \"${value}\"" >> "$OUTPUT_FILE"
    fi
done < "$ENV_FILE"

# Close the enum
cat >> "$OUTPUT_FILE" << 'EOF'
}
EOF

echo "✓ Environment.swift generated successfully"
