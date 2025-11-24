# Configuration Setup

This project uses xcconfig files to manage environment variables securely.

## Setup Instructions

1. Copy the template file:
   ```bash
   cp Config/Secrets.template.xcconfig Config/Secrets.xcconfig
   ```

2. Edit `Config/Secrets.xcconfig` and add your Hugging Face token:
   ```
   HF_TOKEN = hf_your_actual_token_here
   ```

3. The `Secrets.xcconfig` file is in `.gitignore` and will not be committed.

## Using HF_TOKEN in Code

After setting up the xcconfig file and configuring it in Xcode (see below), you can access the token in Swift:

```swift
// Add to Info.plist:
// <key>HF_TOKEN</key>
// <string>$(HF_TOKEN)</string>

// Read in Swift:
if let token = Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String {
    print("HF Token: \(token)")
}
```

## Xcode Configuration

1. In Xcode, select your project in the navigator
2. Select the ChatMeet target
3. Go to Build Settings
4. Search for "Configuration File"
5. Set the Config.xcconfig file for Debug and Release configurations
