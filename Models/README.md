# ChatMeet Models

This directory contains the Core ML model packages used by the ChatMeet application.

## Models

### Whisper Tiny (Speech Recognition)
- **Encoder**: `WhisperEncoder_whisper-tiny.mlpackage`
- **Decoder**: `WhisperDecoder_whisper-tiny.mlpackage`
- **Purpose**: Real-time speech-to-text transcription
- **Source**: OpenAI Whisper model converted to Core ML
- **Size**: ~40 MB each

### Llama 3.2 1B Instruct (Text Summarization)
- **Model**: `StatefulLlama3.2Instruct.mlpackage`
- **Purpose**: Generate meeting summaries in bullet point format
- **Source**: Meta Llama 3.2 1B Instruct converted to Core ML
- **Size**: ~1.2 GB

## Usage

These models are automatically loaded by the application during runtime. The model loading code in `WhisperModel.swift` and `SummarizationModel.swift` will:

1. First check the app bundle (for production builds)
2. Fall back to this `Models/` directory (for development)

## Development

When working in Xcode, ensure these models are **not** copied into the bundle during development to avoid bloating the build. The code is designed to load them directly from this directory during debug builds.

## Git LFS

These large model files should be tracked using Git LFS. See `.gitattributes` for the configuration.

```bash
# Track model packages with Git LFS
Models/**/*.mlpackage/** filter=lfs diff=lfs merge=lfs -text
```

## Model Updates

To update a model:

1. Place the new `.mlpackage` file in this directory
2. Update the model name constant in the corresponding Swift file:
   - `WhisperModel.swift` for Whisper models
   - `SummarizationModel.swift` for the Llama model
3. Test thoroughly before committing
