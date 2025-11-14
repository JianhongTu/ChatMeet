//
//  ParakeetModelService.swift
//  ModelServices
//
//  Transcription model service implementation for Parakeet TDT
//

import Foundation

/// Service wrapper for ParakeetModel implementing the TranscriptionModelProtocol
public class ParakeetModelService: TranscriptionModelProtocol {
    
    // MARK: - TranscriptionModel Protocol
    
    public let modelName: String = "Parakeet TDT v3"
    public let maxContextDuration: TimeInterval = 30.0  // 30 seconds max
    
    public var isLoaded: Bool {
        return parakeetModel.isReady
    }
    
    public let capabilities = ModelCapabilities(
        supportsStreaming: true,   // Parakeet supports token-by-token streaming during decoding
        supportsKVCache: false,     // RNN-T uses LSTM states instead of KV cache
        optimalChunkSize: 30.0,     // Works well with 30s chunks
        requiresFixedLength: true,  // Requires 30s padded input
        recommendedOverlap: 2.0
    )
    
    // MARK: - Internal Implementation
    
    private let parakeetModel: ParakeetModel
    
    public init() {
        self.parakeetModel = ParakeetModel()
    }
    
    public func loadModel() async throws {
        try await parakeetModel.loadModel()
    }
    
    public func unloadModel() {
        parakeetModel.unloadModel()
    }
    
    public func transcribe(
        _ audioChunk: AudioChunk,
        onTokenGenerated: ((String) -> Void)?
    ) async throws -> String {
        guard isLoaded else {
            throw TranscriptionModelError.modelNotLoaded
        }
        
        // Validate duration
        if audioChunk.duration > maxContextDuration {
            throw TranscriptionModelError.audioTooLong(
                duration: audioChunk.duration,
                maxDuration: maxContextDuration
            )
        }
        
        // Convert AudioChunk to WAV data (Parakeet's expected format)
        let wavData = AudioPreprocessor.convertSamplesToWAV(
            audioChunk.samples,
            sampleRate: audioChunk.sampleRate
        )
        
        // Transcribe using the existing Parakeet implementation
        // Pass onTokenGenerated callback for real-time streaming
        do {
            let text = try await parakeetModel.transcribe(wavData, onProgress: onTokenGenerated)
            return text
        } catch {
            throw TranscriptionModelError.inferenceFailed(error.localizedDescription)
        }
    }
    
    /// Set compute backend for the Parakeet model
    /// - Parameter backend: The desired compute backend
    public func setComputeBackend(_ backend: ComputeBackend) async throws {
        try await parakeetModel.setComputeBackend(backend)
    }
}
