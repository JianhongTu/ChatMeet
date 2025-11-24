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
        // Use Task.detached to ensure transcription runs on background thread, not main thread
        try audioSource.start { [weak self] chunk in
            Task.detached {
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
            // WORKAROUND: Ignore progress callbacks during transcription to avoid duplication issues
            // Instead, we'll only update UI with the final deduplicated cumulative text after each chunk
            
            let newText = try await model.transcribe(chunk) { chunkProgressText in
                // Ignore progress updates - we'll send the clean cumulative text after chunk completes
            }
            
            // Deduplicate overlapping content between chunks
            print("StreamingTranscriptionCoordinator: 🔍 Chunk complete. newText: '\(newText)'")
            let deduplicatedText = deduplicateWithPrevious(newText: newText, previousText: cumulativeText)
            
            // Update cumulative text with deduplicated content from this chunk
            if !deduplicatedText.isEmpty {
                if !cumulativeText.isEmpty {
                    cumulativeText += " "
                }
                cumulativeText += deduplicatedText
                
                print("StreamingTranscriptionCoordinator: 🟢 Cumulative text (\(cumulativeText.count) chars): '\(cumulativeText)'")
                
                // Send the clean, deduplicated cumulative text to UI
                Task { @MainActor in
                    self.onUpdate?(self.cumulativeText)
                }
            }
            
        } catch {
            print("StreamingTranscriptionCoordinator: ❌ Chunk processing failed: \(error)")
        }
    }
    
    /// Deduplicate new text by removing overlapping words from the beginning
    /// - Parameters:
    ///   - newText: New transcription from current chunk
    ///   - previousText: Cumulative text from previous chunks
    /// - Returns: Deduplicated new text with overlap removed
    private func deduplicateWithPrevious(newText: String, previousText: String) -> String {
        guard !previousText.isEmpty && !newText.isEmpty else { return newText }
        
        // Split into words
        let previousWords = previousText.split(separator: " ").map(String.init)
        let newWords = newText.split(separator: " ").map(String.init)
        
        guard !previousWords.isEmpty && !newWords.isEmpty else { return newText }
        
        // Normalize for comparison (lowercase, remove punctuation)
        func normalize(_ word: String) -> String {
            return word.lowercased().filter { $0.isLetter || $0.isNumber }
        }
        
        // Find the longest matching suffix of previousWords with prefix of newWords
        var maxOverlap = 0
        let searchWindow = min(20, previousWords.count) // Only check last 20 words
        let startIndex = max(0, previousWords.count - searchWindow)
        
        for i in startIndex..<previousWords.count {
            let suffixWords = previousWords[i...]
            let maxMatch = min(suffixWords.count, newWords.count)
            
            var matchCount = 0
            for j in 0..<maxMatch {
                if normalize(suffixWords[suffixWords.startIndex + j]) == normalize(newWords[j]) {
                    matchCount += 1
                } else {
                    break
                }
            }
            
            if matchCount > maxOverlap {
                maxOverlap = matchCount
            }
        }
        
        // Remove overlapping words from the beginning of new text
        if maxOverlap > 0 {
            let deduplicatedWords = newWords.dropFirst(maxOverlap)
            let result = deduplicatedWords.joined(separator: " ")
            print("StreamingTranscriptionCoordinator: 🔍 Removed \(maxOverlap) overlapping words")
            return result
        }
        
        return newText
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
