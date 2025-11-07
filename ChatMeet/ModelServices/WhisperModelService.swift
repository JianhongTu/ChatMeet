//
//  WhisperModelService.swift
//  ModelServices
//
//  Transcription model service implementation for Whisper
//

import Foundation

/// Service wrapper for WhisperModel implementing the TranscriptionModelProtocol
public class WhisperModelService: TranscriptionModelProtocol {
    
    // MARK: - TranscriptionModel Protocol
    
    public let modelName: String = "Whisper Tiny"
    public let maxContextDuration: TimeInterval = 30.0  // 30 seconds max
    
    public var isLoaded: Bool {
        return whisperModel.isReady
    }
    
    public let capabilities = ModelCapabilities(
        supportsStreaming: true,
        supportsKVCache: true,
        optimalChunkSize: 30.0,
        requiresFixedLength: true,  // Whisper expects 30s fixed input
        recommendedOverlap: 2.0
    )
    
    // MARK: - Internal Implementation
    
    private let whisperModel: WhisperModel
    
    public init() {
        self.whisperModel = WhisperModel()
    }
    
    public func loadModel() async throws {
        try await whisperModel.loadModel()
    }
    
    public func unloadModel() {
        whisperModel.unloadModel()
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
        
        // Convert AudioChunk to WAV data (Whisper's expected format)
        let wavData = AudioPreprocessor.convertSamplesToWAV(
            audioChunk.samples,
            sampleRate: audioChunk.sampleRate
        )
        
        // Transcribe using the existing Whisper implementation
        do {
            let text = try await whisperModel.transcribe(wavData, onProgress: onTokenGenerated)
            return text
        } catch {
            throw TranscriptionModelError.inferenceFailed(error.localizedDescription)
        }
    }
}
