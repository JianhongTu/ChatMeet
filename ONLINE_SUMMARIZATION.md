# Online Summarization Setup

This document explains how to configure and use online API-based summarization in ChatMeet.

## Overview

ChatMeet now supports two summarization modes:
- **On-Device**: Uses the local Llama 3.2 Core ML model (default)
- **Online API**: Uses a remote API endpoint for summarization

## API Configuration

### Default Settings
The app is pre-configured to work with the Nautilus NRP endpoint:
- **Base URL**: `https://ellm.nrp-nautilus.io/v1/chat/completions`
- **Default Model**: `gemma3`

These settings follow the OpenAI-compatible API standard.

### Storing Your API Key Securely

The app uses iOS/macOS Keychain to store API credentials securely:

1. **Open API Settings**:
   - Click the "API Settings" button in the main interface
   
2. **Enter Your Credentials**:
   - **API Key**: Your API authentication key
   - **API Endpoint**: The base URL (defaults to Nautilus)
   - **Model Name**: The model to use (defaults to gemma3)

3. **Save**:
   - Click "Save Credentials"
   - Your API key is encrypted and stored in the device Keychain
   - The key never leaves your device except when making API calls

### Implementation Details

#### Keychain Storage
API credentials are stored using the following service and accounts:
- **Service**: `com.chatmeet.api`
- **Accounts**: 
  - `openai_api_key` - The API key
  - `api_endpoint` - The endpoint URL
  - `model_name` - The model name

#### API Request Format

The app makes requests following the OpenAI-compatible format:

```swift
POST {endpoint}
Headers:
  Authorization: Bearer {api_key}
  Content-Type: application/json

Body:
{
  "model": "gemma3",
  "messages": [
    {"role": "system", "content": "System prompt..."},
    {"role": "user", "content": "User message..."}
  ],
  "temperature": 0.7,
  "max_tokens": 500
}
```

## Using Online Summarization

### Toggle Between Modes

In the main interface:
1. Ensure you have configured your API key (see above)
2. Toggle "Online API Summarization" switch
3. The app will automatically use the online API for all summarization requests

When the toggle is OFF:
- Uses the on-device Llama 3.2 model
- No network calls
- Works offline

When the toggle is ON:
- Uses the configured online API
- Requires internet connection
- May have usage costs depending on your API provider

### Files Modified

The following files implement online summarization:

1. **OnlineSummarizationProvider.swift**: API client for making requests
2. **APIKeyManager.swift**: Secure Keychain storage manager
3. **APISettingsView.swift**: UI for configuring API credentials
4. **SummaryService.swift**: Updated to support both local and online providers
5. **MeetingAssistantViewModel.swift**: Toggle logic for switching providers
6. **MeetingAssistantContentView.swift**: UI controls for the toggle

## Security Considerations

✅ **Secure Storage**: API keys are stored in the system Keychain with `kSecAttrAccessibleWhenUnlocked` protection

✅ **Local Only**: Credentials never sync or leave your device except in API requests

✅ **HTTPS**: All API requests use encrypted HTTPS connections

⚠️ **API Key Safety**: Never commit API keys to source control or share them publicly

## Troubleshooting

### "Please configure API key in settings"
- You need to add your API key in the API Settings before enabling online mode

### "API error: ..."
- Check your API key is valid
- Verify the endpoint URL is correct
- Ensure you have internet connectivity
- Check if your API quota is exceeded

### Toggle is disabled
- API key must be configured first
- Cannot switch during active recording or playback

## Code Example

Programmatically switching providers:

```swift
// Switch to online mode
summaryService.switchToOnlineProvider(
    apiKey: "your-api-key",
    endpoint: "https://ellm.nrp-nautilus.io/v1/chat/completions",
    model: "gemma3"
)

// Switch back to local
summaryService.switchToLocalProvider()
```
