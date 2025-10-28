#!/bin/bash

# Script to load environment variables from .env file
# This should be run as a pre-action script in Xcode scheme

ENV_FILE="${SRCROOT}/.env"

if [ -f "$ENV_FILE" ]; then
    echo "Loading environment variables from .env"
    
    # Read .env file and export variables
    while IFS='=' read -r key value; do
        # Skip comments and empty lines
        if [[ ! "$key" =~ ^#.* ]] && [[ -n "$key" ]]; then
            # Remove quotes from value if present
            value=$(echo "$value" | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")
            # Export the variable
            export "$key=$value"
            echo "Loaded: $key"
        fi
    done < "$ENV_FILE"
else
    echo "Warning: .env file not found at $ENV_FILE"
    echo "Please copy .env.example to .env and add your Hugging Face token"
fi
