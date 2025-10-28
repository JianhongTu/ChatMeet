# ChatMeet

A lightweight iOS and macOS meeting assistant that provides on-device transcription and summarization using Core ML. ChatMeet leverages Whisper for speech-to-text transcription and Llama 3.2 for intelligent summarization, all running locally on your device for maximum privacy.

## Features

- 🎤 **Real-time Audio Recording** - Capture meeting audio with a simple interface
- 🗣️ **On-Device Transcription** - Convert speech to text using Whisper Tiny model
- 📝 **AI Summarization** - Generate bullet-point summaries using Llama 3.2 1B Instruct
- 🔒 **Privacy-First** - All processing happens on your device, no data leaves your machine
- 📱 **Cross-Platform** - Native support for both iOS and macOS with optimized UIs
- ⚡ **Optimized Performance** - CPU-only inference on iOS, GPU acceleration on macOS

## Requirements

- **macOS**: macOS 15.7+ (Sequoia)
- **iOS**: iOS 18.6+ (iPhone 16 recommended for better performance)
- **Xcode**: 16.0+
- **Swift**: 5.0+

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/JianhongTu/ChatMeet.git
cd ChatMeet
```

### 2. Install Git LFS (for model files)

```bash
brew install git-lfs
git lfs install
git lfs pull
```

### 3. Set Up Environment Variables

#### Add your Hugging Face token to Xcode Scheme:

1. In Xcode, click on the scheme selector (next to the run/stop buttons)
2. Select **"Edit Scheme..."** (or press **Cmd+<**)
3. In the left sidebar, select **Run**
4. Go to the **Arguments** tab
5. Under **Environment Variables**, find the `HF_TOKEN` entry
6. Double-click the **Value** field and paste your token from https://huggingface.co/settings/tokens
7. Click **Close**

> **Note:** Each developer needs to add their own token in their local Xcode scheme. The scheme file in git has an empty value for security.

### 4. Build and Run

```bash
# Build from command line
xcodebuild -project ChatMeet.xcodeproj -scheme ChatMeet build

# Or open in Xcode and press Cmd+R to run
open ChatMeet.xcodeproj
```

## Usage

### macOS
1. Launch the app
2. Click "Start Recording" to capture audio
3. Click "Stop Recording" when done
4. View real-time transcription in the left panel
5. AI-generated summary appears in the right panel

### iOS
1. Launch the app
2. Tap "Start Recording" to capture audio
3. Tap "Stop Recording" when done
4. Transcription appears in the top text area
5. Summary appears in the bottom text area

## Architecture

```
ChatMeet/
├── ChatMeet/                    # Main app code
│   ├── ChatMeetApp.swift       # App entry point
│   ├── ContentView.swift       # Root view
│   └── Environment.swift       # Auto-generated (git-ignored)
├── MeetingAssistant/           # UI layer
│   ├── MeetingAssistantContentView.swift  # Main UI (platform-specific)
│   └── MeetingAssistantViewModel.swift    # UI logic
├── MeetingAssistantCore/       # Business logic
│   ├── AudioRecorder.swift     # Audio capture
│   ├── TranscriptionService.swift  # Whisper integration
│   └── SummaryService.swift    # Llama integration
└── MLModels/                   # Core ML models
    ├── WhisperModel.swift      # Whisper wrapper
    ├── SummarizationModel.swift # Llama wrapper
    ├── WhisperEncoder_whisper-tiny.mlpackage
    ├── WhisperDecoder_whisper-tiny.mlpackage
    └── StatefulLlama3.2Instruct.mlpackage
```

## Models

- **Whisper Tiny**: ~75MB, optimized for on-device speech recognition
- **Llama 3.2 1B Instruct**: ~1.5GB, compact language model for summarization

Models are stored using Git LFS and automatically loaded at runtime.

## Technology Stack

- **Swift 5.0** - Primary language
- **SwiftUI** - UI framework
- **Core ML** - Machine learning inference
- **AVFoundation** - Audio recording and playback
- **Hugging Face Transformers** - Model tokenization
- **Git LFS** - Large file storage

## Security & Privacy

- ✅ All processing happens on-device
- ✅ No data sent to external servers
- ✅ Environment variables stored in git-ignored `.env` file
- ✅ Tokens never committed to version control
- ✅ Audio recordings stay local

## Troubleshooting

### Models not loading on iOS
- Models use CPU-only inference on iOS to avoid compatibility issues
- Performance may be slower on older devices
- Recommended: iPhone 16 or newer

### Environment variables not working
- Ensure `.env` file exists in project root
- Check that build phase script is configured correctly
- Clean build folder (Cmd+Shift+K) and rebuild

### Build errors
- Run `git lfs pull` to ensure all model files are downloaded
- Check that all Swift packages are resolved
- Verify Xcode 16+ is installed

For detailed information, see [Dependencies](ADD_DEPENDENCIES.md)

## Contact

For questions or issues, please open an issue on GitHub.
