# How to Load .env File in Xcode

This project uses a **Build Phase Script** to automatically load environment variables from `.env` into your Swift code.

## Quick Setup (3 Steps)

### 1. Create your `.env` file
```bash
cp .env.example .env
# Edit .env and add your actual tokens
```

### 2. Add Build Phase Script in Xcode

1. Open **ChatMeet.xcodeproj** in Xcode
2. Select the **ChatMeet** target (blue icon at top of project navigator)
3. Go to **Build Phases** tab
4. Click **+** button → **New Run Script Phase**
5. Rename it to: `Generate Environment Variables`
6. **Drag it to run BEFORE "Compile Sources"** (very important!)
7. Paste this script:
   ```bash
   "${SRCROOT}/scripts/generate-env.sh"
   ```

8. Click **Input Files** and add:
   ```
   $(SRCROOT)/.env
   ```

9. Click **Output Files** and add:
   ```
   $(SRCROOT)/ChatMeet/Environment.swift
   ```

### 3. Build the project
- Press **Cmd+B** to build
- The script will generate `ChatMeet/Environment.swift`
- You can now use `Environment.HF_TOKEN` in your code

## Using Environment Variables in Your Code

```swift
import Foundation

// Access the token
let token = Environment.HF_TOKEN

// Use it
print("Token loaded: \(token)")
```

## How It Works

```
.env file (git-ignored)
    ↓
Build Phase Script
    ↓
Environment.swift (git-ignored, auto-generated)
    ↓
Your Swift Code
```

## Troubleshooting

**Error: "Environment.swift not found"**
- Make sure you added the Build Phase Script
- Make sure `.env` file exists
- Try building again (Cmd+B)

**Error: ".env file not found"**
- Copy `.env.example` to `.env`
- Make sure it's in the project root directory

**Token not loading correctly**
- Clean build folder (Cmd+Shift+K)
- Rebuild (Cmd+B)
- Check that `Environment.swift` was generated

## Security

✅ `.env` is git-ignored  
✅ `Environment.swift` is git-ignored  
✅ Only `.env.example` and `Environment.swift.template` are in git  
✅ Your secrets stay local and never get committed  

## Alternative: Using Xcode Schemes

If you prefer to use Xcode scheme environment variables instead:
1. Edit scheme (Cmd+<)
2. Select **Run** → **Arguments**
3. Add environment variable: `HF_TOKEN=your_token`
4. Access in code: `ProcessInfo.processInfo.environment["HF_TOKEN"]`

**However, the Build Phase Script approach is recommended** because:
- ✅ Works in all contexts (Run, Test, Archive)
- ✅ Type-safe Swift constants
- ✅ No need to edit scheme per developer
- ✅ Easier to manage multiple environments
