//
//  TranscriptionService.swift
//  MeetingAssistantCore
//
//  Service for transcribing audio to text using Whisper or Parakeet models
//

import Foundation

/// Service that handles audio transcription using the Whisper or Parakeet model
class TranscriptionService: @unchecked Sendable {
    
    private var whisperModel: WhisperModel?
    private var parakeetModel: ParakeetModel?
    private var currentModel: TranscriptionModel?
    
    private var streamingProcessor: StreamingAudioProcessor?
    private var streamingContext: StreamingTranscriptionContext?
    private var isStreaming = false
    private var cumulativeTranscription = ""
    
    // Model readiness states
    private(set) var isTranscriptionModelReady = false
    
    public init() {
        // Models will be loaded lazily when user selects one
    }
    
    /// Switch to a different transcription model
    /// - Parameter model: The model to switch to
    public func switchModel(to model: TranscriptionModel) async throws {
        print("TranscriptionService: 🔄 Switching to \(model.displayName)...")
        
        // Unload current model if any
        whisperModel?.unloadModel()
        parakeetModel?.unloadModel()
        whisperModel = nil
        parakeetModel = nil
        isTranscriptionModelReady = false
        
        // Load the new model
        switch model {
        case .none:
            // No model selected - just keep everything unloaded
            print("TranscriptionService: ⚠️ No model selected")
            
        case .whisperTiny:
            let newWhisper = WhisperModel()
            try await newWhisper.loadModel()
            whisperModel = newWhisper
            isTranscriptionModelReady = newWhisper.isReady
            
        case .parakeetV3:
            let newParakeet = ParakeetModel()
            try await newParakeet.loadModel()
            parakeetModel = newParakeet
            isTranscriptionModelReady = newParakeet.isReady
        }
        
        currentModel = model
        print("TranscriptionService: ✅ Switched to \(model.displayName)")
    }
    
    /// Transcribe audio data to text
    /// - Parameters:
    ///   - audioData: Raw audio data to transcribe (16kHz WAV format)
    ///   - onProgress: Optional callback for real-time token streaming
    /// - Returns: Final transcribed text
    public func transcribe(_ audioData: Data, onProgress: (@Sendable (String) -> Void)? = nil) async throws -> String {
        // Validate audio data
        guard !audioData.isEmpty else {
            throw TranscriptionError.emptyAudioData
        }
        
        // Check minimum audio size (44 byte header + some audio data)
        guard audioData.count > 100 else {
            throw TranscriptionError.audioTooShort
        }
        
        // Check that a model is loaded
        guard isTranscriptionModelReady else {
            throw TranscriptionError.modelNotLoaded
        }
        
        // Process through the selected model with streaming
        do {
            let transcription: String
            
            if let whisper = whisperModel {
                transcription = try await whisper.transcribe(audioData, onProgress: onProgress)
            } else if let parakeet = parakeetModel {
                transcription = try await parakeet.transcribe(audioData, onProgress: onProgress)
            } else {
                throw TranscriptionError.modelNotLoaded
            }
            
            // Validate we got some output
            guard !transcription.isEmpty else {
                throw TranscriptionError.noSpeechDetected
            }
            
            return transcription
            
        } catch let error as WhisperError {
            // Convert Whisper errors to TranscriptionError
            throw TranscriptionError.transcriptionFailed(error.localizedDescription)
        } catch let error as ParakeetError {
            // Convert Parakeet errors to TranscriptionError
            throw TranscriptionError.transcriptionFailed(error.localizedDescription)
        } catch {
            throw TranscriptionError.transcriptionFailed(error.localizedDescription)
        }
    }
    
    /// Transcribe long audio using rolling window strategy
    /// Automatically detects audio length and uses appropriate method
    /// - Parameters:
    ///   - audioData: Raw audio data to transcribe (16kHz WAV format)
    ///   - windowDuration: Duration of each window in seconds (default: 20s)
    ///   - overlapDuration: Overlap between windows in seconds (default: 2s)
    ///   - onProgress: Optional callback for progress updates
    /// - Returns: Complete transcribed text
    public func transcribeLongAudio(
        _ audioData: Data,
        windowDuration: TimeInterval = 20.0,
        overlapDuration: TimeInterval = 2.0,
        onProgress: (@Sendable (String) -> Void)? = nil
    ) async throws -> String {
        // Validate audio data
        guard !audioData.isEmpty else {
            throw TranscriptionError.emptyAudioData
        }
        
        guard audioData.count > 100 else {
            throw TranscriptionError.audioTooShort
        }
        
        guard isTranscriptionModelReady else {
            throw TranscriptionError.modelNotLoaded
        }
        
        // Extract samples to check duration
        let samples = try AudioPreprocessor.extractPCMSamples(from: audioData)
        let duration = Double(samples.count) / 16000.0
        
        print("TranscriptionService: Audio duration: \(String(format: "%.1f", duration))s")
        
        // For audio <= 30s, use regular transcription
        if duration <= 30.0 {
            print("TranscriptionService: Using standard transcription (audio <= 30s)")
            return try await transcribe(audioData, onProgress: onProgress)
        }
        
        // For longer audio, use rolling window
        print("TranscriptionService: Using rolling window transcription (audio > 30s)")
        
        do {
            let transcription: String
            
            if let whisper = whisperModel {
                transcription = try await whisper.transcribeLongAudio(
                    audioData,
                    windowDuration: windowDuration,
                    overlapDuration: overlapDuration,
                    onProgress: onProgress
                )
            } else if let parakeet = parakeetModel {
                transcription = try await parakeet.transcribeLongAudio(
                    audioData,
                    windowDuration: windowDuration,
                    overlapDuration: overlapDuration,
                    onProgress: onProgress
                )
            } else {
                throw TranscriptionError.modelNotLoaded
            }
            
            guard !transcription.isEmpty else {
                throw TranscriptionError.noSpeechDetected
            }
            
            return transcription
            
        } catch let error as WhisperError {
            throw TranscriptionError.transcriptionFailed(error.localizedDescription)
        } catch let error as ParakeetError {
            throw TranscriptionError.transcriptionFailed(error.localizedDescription)
        } catch {
            throw TranscriptionError.transcriptionFailed(error.localizedDescription)
        }
    }
    
    /// Start real-time streaming transcription with sliding window and KV cache reuse
    /// - Parameters:
    ///   - onUpdate: Callback invoked with cumulative transcription updates
    public func startStreamingTranscription(onUpdate: @escaping @Sendable (String) -> Void) throws {
        guard !isStreaming else {
            throw TranscriptionError.alreadyStreaming
        }
        
        guard isTranscriptionModelReady else {
            throw TranscriptionError.modelNotLoaded
        }
        
        // Only Whisper supports streaming for now
        guard whisperModel != nil else {
            throw TranscriptionError.streamingNotSupported
        }
        
        isStreaming = true
        cumulativeTranscription = ""
        
        // Create streaming context to maintain decoder state across chunks
        streamingContext = StreamingTranscriptionContext()
        
        streamingProcessor = StreamingAudioProcessor()
        
        // Start streaming with 5-second chunk processing
        try streamingProcessor?.startStreaming { [weak self] audioChunk in
            guard let self = self, let context = self.streamingContext, let whisper = self.whisperModel else { return }
            
            // Process each chunk incrementally through Whisper
            Task { @MainActor in
                do {
                    let totalContextTokens = context.contextTokens.count
                    
                    // Track the base transcription at start of chunk
                    let baseTranscription = self.cumulativeTranscription
                    let needsSpace = !baseTranscription.isEmpty
                    
                    // Get only NEW text from this chunk with token-by-token streaming
                    let newText = try await whisper.transcribeIncremental(
                        audioData: audioChunk,
                        context: context,
                        onProgress: { partialText in
                            // Stream each token update with cumulative text
                            var streamedText = baseTranscription
                            if needsSpace {
                                streamedText += " "
                            }
                            streamedText += partialText
                            onUpdate(streamedText)
                        }
                    )
                    
                    print("🟢 TranscriptionService: Chunk produced \(newText.count) characters of new text")
                    
                    // Update final cumulative transcription
                    if !newText.isEmpty {
                        // Add space between chunks if there's already content
                        if !self.cumulativeTranscription.isEmpty {
                            self.cumulativeTranscription += " "
                        }
                        self.cumulativeTranscription += newText
                        print("🟢 TranscriptionService: Updated UI with cumulative text (\(self.cumulativeTranscription.count) chars total)")
                    } else {
                        print("⚠️ TranscriptionService: Chunk produced NO new text")
                    }
                    
                } catch {
                    // Continue processing even if one chunk fails
                    print("❌ TranscriptionService: Chunk processing failed: \(error)")
                }
            }
        }
    }
    
    /// Stop real-time streaming transcription
    public func stopStreamingTranscription() {
        guard isStreaming else { return }
        
        streamingProcessor?.stopStreaming()
        streamingProcessor = nil
        streamingContext = nil
        cumulativeTranscription = ""
        isStreaming = false
    }
    
    /// Check if currently streaming
    public var isCurrentlyStreaming: Bool {
        return isStreaming
    }
}

/// Errors that can occur during transcription
public enum TranscriptionError: LocalizedError {
    case emptyAudioData
    case audioTooShort
    case noSpeechDetected
    case modelNotLoaded
    case transcriptionFailed(String)
    case alreadyStreaming
    case streamingNotSupported
    
    public var errorDescription: String? {
        switch self {
        case .emptyAudioData:
            return "No audio data to transcribe"
        case .audioTooShort:
            return "Audio recording is too short"
        case .noSpeechDetected:
            return "No speech detected in the audio"
        case .modelNotLoaded:
            return "Transcription model is not loaded"
        case .transcriptionFailed(let reason):
            return "Transcription failed: \(reason)"
        case .alreadyStreaming:
            return "Streaming transcription is already active"
        case .streamingNotSupported:
            return "Current model does not support streaming"
        }
    }
}
