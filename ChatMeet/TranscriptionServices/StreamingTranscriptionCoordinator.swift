//
//  StreamingTranscriptionCoordinator.swift
//  TranscriptionServices
//
//  Coordinates real-time streaming transcription
//

import Foundation

/// Coordinator for real-time streaming transcription
public class StreamingTranscriptionCoordinator {
    
    // MARK: - Properties
    
    private var model: TranscriptionModelProtocol?
    private var audioSource: AudioSource?
    private var context: StreamingTranscriptionContext?
    
    private var isStreaming = false
    private var cumulativeText = ""
    private var onUpdate: (@Sendable (String) -> Void)?
    
    // MARK: - Initialization
    
    public init() {}
    
    // MARK: - Streaming Control
    
    /// Start streaming transcription
    /// - Parameters:
    ///   - model: The transcription model to use (must support streaming)
    ///   - audioSource: Source of audio chunks
    ///   - onUpdate: Callback invoked with cumulative transcription updates
    public func startStreaming(
        model: TranscriptionModelProtocol,
        audioSource: AudioSource,
        onUpdate: @escaping @Sendable (String) -> Void
    ) throws {
        guard !isStreaming else {
            throw StreamingTranscriptionError.alreadyStreaming
        }
        
        // Verify model supports streaming
        guard model.capabilities.supportsStreaming else {
            throw StreamingTranscriptionError.modelDoesNotSupportStreaming
        }
        
        guard model.isLoaded else {
            throw StreamingTranscriptionError.modelNotLoaded
        }
        
        self.model = model
        self.audioSource = audioSource
        self.onUpdate = onUpdate
        self.cumulativeText = ""
        self.context = StreamingTranscriptionContext()
        self.isStreaming = true
        
        print("StreamingTranscriptionCoordinator: ✅ Starting streaming with \(model.modelName)")
        
        // Start audio source with chunk processing
        try audioSource.start { [weak self] chunk in
            Task {
                await self?.processChunk(chunk)
            }
        }
    }
    
    /// Stop streaming transcription
    public func stopStreaming() {
        guard isStreaming else { return }
        
        audioSource?.stop()
        audioSource = nil
        model = nil
        context = nil
        onUpdate = nil
        isStreaming = false
        
        print("StreamingTranscriptionCoordinator: 🛑 Stopped streaming")
    }
    
    /// Check if currently streaming
    public var isCurrentlyStreaming: Bool {
        return isStreaming
    }
    
    /// Get accumulated transcription
    public var transcription: String {
        return cumulativeText
    }
    
    // MARK: - Private Methods
    
    private func processChunk(_ chunk: AudioChunk) async {
        guard let model = model, let context = context else { return }
        
        do {
            // Track base text before this chunk
            let baseText = cumulativeText
            let needsSpace = !baseText.isEmpty
            
            // For streaming, we need to use the model's streaming capabilities
            // This requires access to the underlying WhisperModel's transcribeIncremental method
            // For now, we'll use the standard transcribe method with streaming callback
            
            var partialText = ""
            let newText = try await model.transcribe(chunk) { token in
                partialText += token
                
                // Stream incremental updates
                var streamedText = baseText
                if needsSpace && !streamedText.isEmpty {
                    streamedText += " "
                }
                streamedText += partialText
                
                Task { @MainActor in
                    self.onUpdate?(streamedText)
                }
            }
            
            // Update cumulative text with new content
            if !newText.isEmpty {
                if !cumulativeText.isEmpty {
                    cumulativeText += " "
                }
                cumulativeText += newText
                
                print("StreamingTranscriptionCoordinator: 🟢 Chunk produced \(newText.count) chars")
                
                // Final update for this chunk
                Task { @MainActor in
                    self.onUpdate?(self.cumulativeText)
                }
            }
            
        } catch {
            print("StreamingTranscriptionCoordinator: ❌ Chunk processing failed: \(error)")
        }
    }
}

/// Errors for streaming transcription
public enum StreamingTranscriptionError: LocalizedError {
    case alreadyStreaming
    case modelDoesNotSupportStreaming
    case modelNotLoaded
    case processingFailed(String)
    
    public var errorDescription: String? {
        switch self {
        case .alreadyStreaming:
            return "Streaming transcription is already active"
        case .modelDoesNotSupportStreaming:
            return "Selected model does not support streaming transcription"
        case .modelNotLoaded:
            return "Transcription model is not loaded"
        case .processingFailed(let reason):
            return "Streaming processing failed: \(reason)"
        }
    }
}
