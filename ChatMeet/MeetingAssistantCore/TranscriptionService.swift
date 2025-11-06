//
//  TranscriptionService.swift
//  MeetingAssistantCore
//
//  Service for transcribing audio to text using Whisper
//

import Foundation

/// Service that handles audio transcription using the Whisper model
class TranscriptionService: @unchecked Sendable {
    
    private let whisperModel: WhisperModel
    private var streamingProcessor: StreamingAudioProcessor?
    private var streamingContext: StreamingTranscriptionContext?
    private var isStreaming = false
    private var cumulativeTranscription = ""
    
    public init() {
        self.whisperModel = WhisperModel()
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
        
        // Process through Whisper model with streaming
        do {
            let transcription = try await whisperModel.transcribe(audioData, onProgress: onProgress)
            
            // Validate we got some output
            guard !transcription.isEmpty else {
                throw TranscriptionError.noSpeechDetected
            }
            
            return transcription
            
        } catch let error as WhisperError {
            // Convert Whisper errors to TranscriptionError
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
        
        isStreaming = true
        cumulativeTranscription = ""
        
        // Create streaming context to maintain decoder state across chunks
        streamingContext = StreamingTranscriptionContext()
        
        streamingProcessor = StreamingAudioProcessor()
        
        // Start streaming with 5-second chunk processing
        try streamingProcessor?.startStreaming { [weak self] audioChunk in
            guard let self = self, let context = self.streamingContext else { return }
            
            // Process each chunk incrementally through Whisper
            Task { @MainActor in
                do {
                    let totalContextTokens = context.contextTokens.count
                    
                    // Track the base transcription at start of chunk
                    let baseTranscription = self.cumulativeTranscription
                    let needsSpace = !baseTranscription.isEmpty
                    
                    // Get only NEW text from this chunk with token-by-token streaming
                    let newText = try await self.whisperModel.transcribeIncremental(
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
    
    public var errorDescription: String? {
        switch self {
        case .emptyAudioData:
            return "No audio data to transcribe"
        case .audioTooShort:
            return "Audio recording is too short"
        case .noSpeechDetected:
            return "No speech detected in the audio"
        case .modelNotLoaded:
            return "Whisper model is not loaded"
        case .transcriptionFailed(let reason):
            return "Transcription failed: \(reason)"
        case .alreadyStreaming:
            return "Streaming transcription is already active"
        }
    }
}
