//
//  TranscriptionService.swift
//  MeetingAssistantCore
//
//  Service for transcribing audio to text using Whisper or Parakeet models
//  Refactored to use the new TranscriptionCoordinator architecture
//

import Foundation

/// Service that handles audio transcription using the Whisper or Parakeet model
class TranscriptionService: @unchecked Sendable {
    
    // New architecture components
    private let coordinator = TranscriptionCoordinator()
    private let streamingCoordinator = StreamingTranscriptionCoordinator()
    
    // Track current model selection
    private var currentModelType: TranscriptionModel?
    private var currentModelService: TranscriptionModelProtocol?
    
    // Model readiness states
    private(set) var isTranscriptionModelReady = false
    
    public init() {
        // Models will be loaded lazily when user selects one
    }
    
    /// Switch to a different transcription model
    /// - Parameter model: The model to switch to
    public func switchModel(to model: TranscriptionModel) async throws {
        print("TranscriptionService: 🔄 Switching to \(model.displayName)...")
        
        // Unload current coordinators
        coordinator.unloadModel()
        streamingCoordinator.stopStreaming()
        
        currentModelService = nil
        isTranscriptionModelReady = false
        
        // Load the new model through coordinator
        switch model {
        case .none:
            // No model selected - just keep everything unloaded
            print("TranscriptionService: ⚠️ No model selected")
            
        case .whisperTiny:
            // Load through new architecture
            let modelService = WhisperModelService()
            try await coordinator.loadModel(modelService)
            currentModelService = modelService
            isTranscriptionModelReady = coordinator.isReady
            
        case .parakeetV3:
            // Load through new architecture
            let modelService = ParakeetModelService()
            try await coordinator.loadModel(modelService)
            currentModelService = modelService
            isTranscriptionModelReady = coordinator.isReady
        }
        
        currentModelType = model
        print("TranscriptionService: ✅ Switched to \(model.displayName)")
    }
    
    /// Switch compute backend for the current model
    /// - Parameter backend: The compute backend to use
    public func switchBackend(to backend: ComputeBackend) async throws {
        guard let modelService = currentModelService else {
            throw TranscriptionError.modelNotLoaded
        }
        
        print("TranscriptionService: 🔄 Switching backend to \(backend.description)...")
        
        // Switch backend based on model type
        if let whisperService = modelService as? WhisperModelService {
            try await whisperService.setComputeBackend(backend)
        } else if let parakeetService = modelService as? ParakeetModelService {
            try await parakeetService.setComputeBackend(backend)
        }
        
        print("TranscriptionService: ✅ Backend switched to \(backend.description)")
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
        
        // Convert Data to AudioData
        let samples = try AudioPreprocessor.extractPCMSamples(from: audioData)
        let audio = AudioData(
            samples: samples,
            sampleRate: 16000,
            format: .wav
        )
        
        // Use the new coordinator with auto mode for short audio
        do {
            let result = try await coordinator.transcribe(audio, mode: .auto) { progress in
                if let onProgress = onProgress {
                    onProgress(progress.partialText)
                }
            }
            
            // Validate we got some output
            guard !result.text.isEmpty else {
                throw TranscriptionError.noSpeechDetected
            }
            
            return result.text
            
        } catch let error as TranscriptionCoordinatorError {
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
        
        // Convert Data to AudioData
        let samples = try AudioPreprocessor.extractPCMSamples(from: audioData)
        let audio = AudioData(
            samples: samples,
            sampleRate: 16000,
            format: .wav
        )
        
        print("TranscriptionService: Audio duration: \(String(format: "%.1f", audio.duration))s")
        
        // Use the new coordinator with batch mode for long audio
        let mode: TranscriptionMode = audio.duration > 30.0 
            ? .batch(maxWindow: windowDuration) 
            : .auto
        
        print("TranscriptionService: Using \(mode) mode")
        
        do {
            let result = try await coordinator.transcribe(audio, mode: mode) { progress in
                if let onProgress = onProgress {
                    onProgress(progress.partialText)
                }
            }
            
            guard !result.text.isEmpty else {
                throw TranscriptionError.noSpeechDetected
            }
            
            return result.text
            
        } catch let error as TranscriptionCoordinatorError {
            throw TranscriptionError.transcriptionFailed(error.localizedDescription)
        } catch {
            throw TranscriptionError.transcriptionFailed(error.localizedDescription)
        }
    }
    
    /// Start real-time streaming transcription with sliding window and KV cache reuse
    /// - Parameters:
    ///   - onUpdate: Callback invoked with cumulative transcription updates
    public func startStreamingTranscription(onUpdate: @escaping @Sendable (String) -> Void) throws {
        guard !streamingCoordinator.isCurrentlyStreaming else {
            throw TranscriptionError.alreadyStreaming
        }
        
        guard isTranscriptionModelReady else {
            throw TranscriptionError.modelNotLoaded
        }
        
        guard let modelService = currentModelService else {
            throw TranscriptionError.modelNotLoaded
        }
        
        // Verify model supports streaming
        guard modelService.capabilities.supportsStreaming else {
            throw TranscriptionError.streamingNotSupported
        }
        
        // Create live audio source
        let audioSource = LiveAudioSource(
            chunkDuration: 5.0,
            sampleRate: 16000,
            maxBufferDuration: 30.0
        )
        
        // Start streaming through coordinator
        try streamingCoordinator.startStreaming(
            model: modelService,
            audioSource: audioSource,
            onUpdate: onUpdate
        )
        
        print("TranscriptionService: ✅ Started streaming transcription")
    }
    
    /// Stop real-time streaming transcription
    public func stopStreamingTranscription() {
        streamingCoordinator.stopStreaming()
        print("TranscriptionService: 🛑 Stopped streaming transcription")
    }
    
    /// Check if currently streaming
    public var isCurrentlyStreaming: Bool {
        return streamingCoordinator.isCurrentlyStreaming
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
