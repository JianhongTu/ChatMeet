//
//  TranscriptionCoordinator.swift
//  TranscriptionServices
//
//  Coordinates transcription across models and strategies
//

import Foundation

/// Main service for coordinating transcription operations
public class TranscriptionCoordinator {
    
    // MARK: - Properties
    
    private var currentModel: TranscriptionModelProtocol?
    private var currentStrategy: ContextWindowStrategy?
    
    /// Check if a model is loaded and ready
    public var isReady: Bool {
        return currentModel?.isLoaded ?? false
    }
    
    /// Current model name
    public var modelName: String? {
        return currentModel?.modelName
    }
    
    // MARK: - Initialization
    
    public init() {
        // Start with no model loaded
    }
    
    // MARK: - Model Management
    
    /// Load a specific transcription model
    /// - Parameter model: The model to load
    public func loadModel(_ model: TranscriptionModelProtocol) async throws {
        // Unload current model if any
        currentModel?.unloadModel()
        
        // Load new model
        try await model.loadModel()
        currentModel = model
        
        print("TranscriptionCoordinator: ✅ Loaded model: \(model.modelName)")
    }
    
    /// Unload the current model
    public func unloadModel() {
        currentModel?.unloadModel()
        currentModel = nil
        currentStrategy = nil
        print("TranscriptionCoordinator: 🗑️ Model unloaded")
    }
    
    // MARK: - Transcription
    
    /// Unified transcription method
    /// - Parameters:
    ///   - audio: The audio data to transcribe
    ///   - mode: Transcription mode (streaming, batch, or auto)
    ///   - onProgress: Optional progress callback
    /// - Returns: Transcription result with text and metadata
    public func transcribe(
        _ audio: AudioData,
        mode: TranscriptionMode = .auto,
        onProgress: ((TranscriptionProgress) -> Void)? = nil
    ) async throws -> TranscriptionResult {
        guard let model = currentModel else {
            throw TranscriptionCoordinatorError.noModelLoaded
        }
        
        let startTime = Date()
        
        // Determine strategy based on mode
        let strategy = try selectStrategy(for: audio, mode: mode, modelCapabilities: model.capabilities)
        self.currentStrategy = strategy
        
        // Chunk the audio
        let chunks = strategy.chunkAudio(audio)
        
        print("TranscriptionCoordinator: Processing \(chunks.count) chunks")
        
        // Transcribe each chunk
        var segments: [TranscriptionSegment] = []
        var processedDuration: TimeInterval = 0
        
        for (index, chunk) in chunks.enumerated() {
            // Progress callback
            if let onProgress = onProgress {
                let progress = TranscriptionProgress(
                    processedDuration: processedDuration,
                    totalDuration: audio.duration,
                    partialText: strategy.mergeTranscriptions(segments),
                    confidence: nil
                )
                onProgress(progress)
            }
            
            // Transcribe chunk
            var partialText = ""
            let chunkText = try await model.transcribe(chunk) { token in
                partialText += token
                // Stream partial tokens through progress callback
                if let onProgress = onProgress {
                    let progress = TranscriptionProgress(
                        processedDuration: processedDuration,
                        totalDuration: audio.duration,
                        partialText: strategy.mergeTranscriptions(segments) + " " + partialText,
                        confidence: nil
                    )
                    onProgress(progress)
                }
            }
            
            // Create segment
            let segment = TranscriptionSegment(
                text: chunkText,
                startTime: chunk.startTime,
                endTime: chunk.endTime,
                confidence: nil
            )
            segments.append(segment)
            
            processedDuration = chunk.endTime
            
            print("TranscriptionCoordinator: Chunk \(index + 1)/\(chunks.count) complete")
        }
        
        // Merge segments
        let finalText = strategy.mergeTranscriptions(segments)
        
        // Create metadata
        let processingTime = Date().timeIntervalSince(startTime)
        let metadata = TranscriptionMetadata(
            modelName: model.modelName,
            sampleRate: audio.sampleRate,
            audioDuration: audio.duration,
            processingTime: processingTime,
            mode: mode
        )
        
        // Final progress callback
        if let onProgress = onProgress {
            let progress = TranscriptionProgress(
                processedDuration: audio.duration,
                totalDuration: audio.duration,
                partialText: finalText,
                confidence: nil
            )
            onProgress(progress)
        }
        
        return TranscriptionResult(
            text: finalText,
            segments: segments,
            metadata: metadata
        )
    }
    
    // MARK: - Strategy Selection
    
    private func selectStrategy(
        for audio: AudioData,
        mode: TranscriptionMode,
        modelCapabilities: ModelCapabilities
    ) throws -> ContextWindowStrategy {
        
        switch mode {
        case .streaming(let chunkSize):
            // Use streaming strategy with specified chunk size
            let overlap = modelCapabilities.recommendedOverlap
            return StreamingContextStrategy(windowSize: chunkSize, overlapDuration: overlap)
            
        case .batch(let maxWindow):
            // Use batch strategy with specified window
            let overlap = modelCapabilities.recommendedOverlap
            return BatchContextStrategy(windowSize: maxWindow, overlapDuration: overlap)
            
        case .auto:
            // Automatically decide based on audio duration and model capabilities
            if audio.duration <= modelCapabilities.optimalChunkSize {
                // Short audio: use single chunk
                return BatchContextStrategy(
                    windowSize: modelCapabilities.optimalChunkSize,
                    overlapDuration: 0
                )
            } else if audio.duration > 60.0 {
                // Long audio: use batch strategy
                return BatchContextStrategy(
                    windowSize: modelCapabilities.optimalChunkSize,
                    overlapDuration: modelCapabilities.recommendedOverlap
                )
            } else {
                // Medium audio: use streaming strategy for responsiveness
                return StreamingContextStrategy(
                    windowSize: modelCapabilities.optimalChunkSize / 2,
                    overlapDuration: modelCapabilities.recommendedOverlap
                )
            }
        }
    }
}

/// Errors specific to the transcription coordinator
public enum TranscriptionCoordinatorError: LocalizedError {
    case noModelLoaded
    case invalidAudioData
    case strategySelectionFailed
    
    public var errorDescription: String? {
        switch self {
        case .noModelLoaded:
            return "No transcription model is loaded"
        case .invalidAudioData:
            return "Invalid audio data provided"
        case .strategySelectionFailed:
            return "Failed to select appropriate chunking strategy"
        }
    }
}
