//
//  StreamingTranscriptionCoordinator.swift
//  TranscriptionServices
//
//  Coordinates real-time streaming transcription with token-level merging
//

import Foundation
import SwiftUI

/// Coordinator for real-time streaming transcription
public class StreamingTranscriptionCoordinator {
    
    // MARK: - Properties
    
    private var model: TranscriptionModelProtocol?
    private var audioSource: AudioSource?
    private var context: StreamingTranscriptionContext?
    
    private var isStreaming = false
    private var cumulativeText = ""
    private var hypothesisText = ""  // Track hypothesis separately
    private var onUpdate: (@Sendable (String) -> Void)?
    private var onAttributedUpdate: (@Sendable (AttributedString) -> Void)?  // New callback for styled text
    
    // Token-level tracking for frame-aligned merging
    private var confirmationTokens: [TokenWindow] = []  // Accurate final tokens (accumulated)
    private let tokenMerger = TokenMerger()
    private let confirmationOverlap: Double = 2.0  // 2s overlap for confirmation
    private let samplesPerFrame: Int = 1280  // Parakeet encoder frame size (80ms @ 16kHz)
    
    // MARK: - Initialization
    
    public init() {}
    
    // MARK: - Streaming Control
    
    /// Start streaming transcription
    /// - Parameters:
    ///   - model: The transcription model to use (must support streaming)
    ///   - audioSource: Source of audio chunks
    ///   - onUpdate: Callback invoked with cumulative transcription updates (plain text)
    ///   - onAttributedUpdate: Optional callback for styled text with gray hypothesis
    public func startStreaming(
        model: TranscriptionModelProtocol,
        audioSource: AudioSource,
        onUpdate: @escaping @Sendable (String) -> Void,
        onAttributedUpdate: (@Sendable (AttributedString) -> Void)? = nil
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
        self.onAttributedUpdate = onAttributedUpdate
        self.cumulativeText = ""
        self.hypothesisText = ""
        self.context = StreamingTranscriptionContext()
        self.isStreaming = true
        
        print("StreamingTranscriptionCoordinator: ✅ Starting dual-mode streaming with \(model.modelName)")
        
        // Start audio source with dual-mode chunk processing
        if let liveSource = audioSource as? LiveAudioSource {
            try liveSource.startDualMode(
                onHypothesis: { [weak self] chunk in
                    Task.detached {
                        await self?.processHypothesisChunk(chunk)
                    }
                },
                onConfirmation: { [weak self] chunk in
                    Task.detached {
                        await self?.processConfirmationChunk(chunk)
                    }
                }
            )
        } else {
            // Fallback to single mode
            try audioSource.start { [weak self] chunk in
                Task.detached {
                    await self?.processConfirmationChunk(chunk)
                }
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
        onAttributedUpdate = nil
        isStreaming = false
        confirmationTokens = []
        hypothesisText = ""
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
    
    /// Process fast hypothesis chunks (cumulative) for low-latency preview
    /// Hypothesis shows cumulative preview of ALL NEW audio after confirmation
    private func processHypothesisChunk(_ chunk: AudioChunk) async {
        guard let model = model as? ParakeetModelService else { return }
        
        do {
            let (tokens, tokenStartFrames, confidences) = try await model.transcribeWithTokens(chunk) { _ in }
            
            // Skip if no tokens generated
            guard !tokens.isEmpty else {
                return
            }
            
            // Decode the cumulative hypothesis chunk (all audio after confirmation)
            let cumulativeHypothesis = model.decodeTokens(tokens)
            hypothesisText = cumulativeHypothesis
            
            // Combine with confirmation base (if exists) or show standalone
            let displayText: String
            if !confirmationTokens.isEmpty {
                // Append hypothesis to confirmation base
                let confirmationBase = model.decodeTokens(confirmationTokens.map { $0.token })
                displayText = confirmationBase + " " + cumulativeHypothesis
                print("🔵 HYPOTHESIS (cumulative): \"\(cumulativeHypothesis)\"")
            } else {
                // No confirmation yet, show just this hypothesis
                displayText = cumulativeHypothesis
                print("🔵 HYPOTHESIS (cumulative): \"\(cumulativeHypothesis)\"")
            }
            
            // Send updates to UI
            Task { @MainActor in
                self.onUpdate?(displayText)
                
                // Send attributed string with gray hypothesis if callback is set
                if let onAttributedUpdate = self.onAttributedUpdate {
                    var attributed = AttributedString(cumulativeText)
                    if !cumulativeText.isEmpty && !cumulativeHypothesis.isEmpty {
                        attributed.append(AttributedString(" "))
                    }
                    
                    var grayHypothesis = AttributedString(cumulativeHypothesis)
                    #if os(macOS)
                    grayHypothesis.foregroundColor = .gray
                    #else
                    grayHypothesis.foregroundColor = .gray
                    #endif
                    attributed.append(grayHypothesis)
                    
                    onAttributedUpdate(attributed)
                }
            }
            
        } catch {
            print("❌ Hypothesis error: \(error)")
        }
    }
    
    /// Process accurate confirmation chunks (15s) for final transcription
    /// Confirmation is the authoritative accumulated transcript
    private func processConfirmationChunk(_ chunk: AudioChunk) async {
        guard let model = model as? ParakeetModelService else { return }
        
        do {
            let (tokens, tokenStartFrames, confidences) = try await model.transcribeWithTokens(chunk) { _ in }
            
            // Convert to TokenWindow with absolute frame positions
            // Use chunk's startTime to determine absolute position in the stream
            let chunkStartSamples = Int(chunk.startTime * 16000.0)  // Convert startTime to samples
            var newTokenWindows: [TokenWindow] = []
            for i in 0..<tokens.count {
                // Convert relative frame to absolute frame using chunk start position
                let relativeFrameSamples = tokenStartFrames[i] * samplesPerFrame
                let absoluteSamples = chunkStartSamples + relativeFrameSamples
                let absoluteFrame = absoluteSamples / samplesPerFrame
                
                newTokenWindows.append(TokenWindow(
                    token: tokens[i],
                    timestamp: absoluteFrame,
                    confidence: confidences[i]
                ))
            }
            
            let newChunkText = model.decodeTokens(tokens)
            
            // Skip if no tokens generated
            guard !newTokenWindows.isEmpty else {
                return
            }
            
            // With VAD-driven segments, each confirmation is a complete segment (B→E)
            // Segments are separated by silence, so should NEVER overlap - always concatenate
            let mergedTokens: [TokenWindow]
            if confirmationTokens.isEmpty {
                mergedTokens = newTokenWindows
                print("🟢 CONFIRMATION: \"\(newChunkText)\" [\(newTokenWindows.count) tokens]")
            } else {
                // VAD segments are always sequential (separated by silence)
                // Just concatenate - no merging needed
                mergedTokens = confirmationTokens + newTokenWindows
                
                let lastConfirmedFrame = confirmationTokens.last!.timestamp
                let firstNewFrame = newTokenWindows.first!.timestamp
                let gap = firstNewFrame - lastConfirmedFrame
                
                print("🟢 CONFIRMATION: \"\(newChunkText)\" [\(newTokenWindows.count) tokens]")
                print("➕ CONCATENATE: prev(\(confirmationTokens.count)) + new(\(newTokenWindows.count)) = \(mergedTokens.count) tokens, gap=\(gap) frames")
            }
            
            confirmationTokens = mergedTokens
            
            // VAD-driven: Don't use fixed stride, just track absolute positions from timestamps
            // The sample offset is already being updated correctly by the timestamp conversion
            
            // Display confirmation text
            let tokenIds = mergedTokens.map { $0.token }
            let confirmationText = model.decodeTokens(tokenIds)
            cumulativeText = confirmationText
            hypothesisText = ""  // Clear hypothesis after confirmation
            
            print("📝 RESULT: \"\(confirmationText)\"\n")
            
            // Send confirmation update to UI
            Task { @MainActor in
                self.onUpdate?(confirmationText)
                
                // Send attributed string (just black text, no gray hypothesis)
                if let onAttributedUpdate = self.onAttributedUpdate {
                    let attributed = AttributedString(confirmationText)
                    onAttributedUpdate(attributed)
                }
            }
            
        } catch {
            print("❌ Confirmation error: \(error)")
        }
    }
    
    /// Fallback method for models that don't support token-level API
    private func processChunkWithTextFallback(_ chunk: AudioChunk, model: TranscriptionModelProtocol, context: StreamingTranscriptionContext) async {
        do {
            let newText = try await model.transcribe(chunk) { chunkProgressText in
                // Ignore progress updates
            }
            
            // Use old word-based deduplication
            print("StreamingTranscriptionCoordinator: 🔍 Chunk complete (text fallback). newText: '\(newText)'")
            let deduplicatedText = deduplicateWithPrevious(newText: newText, previousText: cumulativeText)
            
            if !deduplicatedText.isEmpty {
                if !cumulativeText.isEmpty {
                    cumulativeText += " "
                }
                cumulativeText += deduplicatedText
                
                print("StreamingTranscriptionCoordinator: 🟢 Cumulative text (\(cumulativeText.count) chars): '\(cumulativeText)'")
                
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
