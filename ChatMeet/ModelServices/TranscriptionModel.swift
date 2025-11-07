//
//  TranscriptionModel.swift
//  ModelServices
//
//  Protocol defining the interface for transcription model implementations
//

import Foundation

/// Protocol for transcription model implementations (renamed to avoid conflict with enum)
public protocol TranscriptionModelProtocol {
    /// Unique name of the model
    var modelName: String { get }
    
    /// Maximum audio duration this model can process at once (in seconds)
    var maxContextDuration: TimeInterval { get }
    
    /// Whether the model is loaded and ready
    var isLoaded: Bool { get }
    
    /// Model capabilities
    var capabilities: ModelCapabilities { get }
    
    /// Load the model into memory
    func loadModel() async throws
    
    /// Unload the model from memory
    func unloadModel()
    
    /// Transcribe a single audio chunk
    /// - Parameters:
    ///   - audioChunk: The audio chunk to transcribe
    ///   - onTokenGenerated: Optional callback for streaming tokens as they're generated
    /// - Returns: Transcribed text
    func transcribe(
        _ audioChunk: AudioChunk,
        onTokenGenerated: ((String) -> Void)?
    ) async throws -> String
}

/// Model capabilities for optimization decisions
public struct ModelCapabilities {
    /// Can generate tokens incrementally
    public let supportsStreaming: Bool
    
    /// Supports KV cache for faster incremental decoding
    public let supportsKVCache: Bool
    
    /// Optimal chunk size for this model (in seconds)
    public let optimalChunkSize: TimeInterval
    
    /// Whether model requires fixed-length input
    public let requiresFixedLength: Bool
    
    /// Recommended overlap between chunks (in seconds)
    public let recommendedOverlap: TimeInterval
    
    public init(
        supportsStreaming: Bool,
        supportsKVCache: Bool,
        optimalChunkSize: TimeInterval,
        requiresFixedLength: Bool,
        recommendedOverlap: TimeInterval
    ) {
        self.supportsStreaming = supportsStreaming
        self.supportsKVCache = supportsKVCache
        self.optimalChunkSize = optimalChunkSize
        self.requiresFixedLength = requiresFixedLength
        self.recommendedOverlap = recommendedOverlap
    }
}

/// Errors that can occur during model operations
public enum TranscriptionModelError: LocalizedError {
    case modelNotLoaded
    case modelLoadFailed(String)
    case audioTooLong(duration: TimeInterval, maxDuration: TimeInterval)
    case invalidAudioFormat
    case inferenceFailed(String)
    
    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "Model is not loaded"
        case .modelLoadFailed(let reason):
            return "Failed to load model: \(reason)"
        case .audioTooLong(let duration, let maxDuration):
            return "Audio duration (\(duration)s) exceeds maximum (\(maxDuration)s)"
        case .invalidAudioFormat:
            return "Invalid audio format for this model"
        case .inferenceFailed(let reason):
            return "Inference failed: \(reason)"
        }
    }
}
