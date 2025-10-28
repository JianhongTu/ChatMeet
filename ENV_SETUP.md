# Environment Setup

This project uses environment variables to store sensitive configuration like API tokens.

## Setup Instructions

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Add your Hugging Face token:**
   - Get your token from: https://huggingface.co/settings/tokens
   - Open `.env` and replace `your_token_here` with your actual token:
     ```
     HF_TOKEN=hf_your_actual_token_here
     ```

3. **The `.env` file is git-ignored:**
   - Your token will never be committed to the repository
   - Each developer needs their own `.env` file

## How It Works

- The `.env` file stores your local configuration
- The `scripts/load-env.sh` script reads the `.env` file
- Xcode runs this script before launching the app (configured in the scheme)
- Environment variables are available to your app at runtime

## Security Notes

⚠️ **Never commit your `.env` file to git!**
- Always use `.env.example` to share the structure
- Keep your tokens private
- Rotate tokens if they're accidentally exposed
