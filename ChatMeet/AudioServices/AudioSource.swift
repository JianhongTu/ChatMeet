//
//  AudioSource.swift
//  AudioServices
//
//  Protocol for audio input sources (live recording, file playback, etc.)
//

import Foundation

/// Protocol for sources that provide audio data
public protocol AudioSource {
    /// Start capturing/providing audio
    /// - Parameter onChunk: Callback invoked with each audio chunk
    func start(onChunk: @escaping @Sendable (AudioChunk) -> Void) throws
    
    /// Start streaming with dual-mode chunks (hypothesis + confirmation)
    /// - Parameters:
    ///   - onHypothesis: Fast 3s chunks for low-latency preview
    ///   - onConfirmation: Accurate 15s chunks for final transcription
    func startDualMode(
        onHypothesis: @escaping @Sendable (AudioChunk) -> Void,
        onConfirmation: @escaping @Sendable (AudioChunk) -> Void
    ) throws
    
    /// Stop capturing/providing audio
    func stop()
    
    /// Check if currently active
    var isActive: Bool { get }
}

/// Live microphone audio source for streaming transcription
public class LiveAudioSource: NSObject, AudioSource, @unchecked Sendable {
    
    // Configuration
    private let chunkDuration: TimeInterval
    private let sampleRate: Int
    private let maxBufferDuration: TimeInterval
    
    // Underlying processor
    private var processor: StreamingAudioProcessor?
    
    // State
    private var _isActive = false
    private var startTime: Date?
    
    public var isActive: Bool {
        return _isActive
    }
    
    /// Initialize live audio source
    /// - Parameters:
    ///   - chunkDuration: Duration of each audio chunk (default: 5s)
    ///   - sampleRate: Sample rate for audio (default: 16000)
    ///   - maxBufferDuration: Maximum buffer duration (default: 30s)
    public init(
        chunkDuration: TimeInterval = 5.0,
        sampleRate: Int = 16000,
        maxBufferDuration: TimeInterval = 30.0
    ) {
        self.chunkDuration = chunkDuration
        self.sampleRate = sampleRate
        self.maxBufferDuration = maxBufferDuration
        super.init()
    }
    
    public func start(onChunk: @escaping @Sendable (AudioChunk) -> Void) throws {
        guard !_isActive else { return }
        
        _isActive = true
        startTime = Date()
        processor = StreamingAudioProcessor()
        
        // Use dual mode with same callback for both (legacy compatibility)
        try processor?.startStreaming(
            onHypothesis: { [weak self] audioData, startSample in
                self?.convertAndEmit(audioData: audioData, startSample: startSample, onChunk: onChunk)
            },
            onConfirmation: { [weak self] audioData, startSample in
                self?.convertAndEmit(audioData: audioData, startSample: startSample, onChunk: onChunk)
            }
        )
    }
    
    public func startDualMode(
        onHypothesis: @escaping @Sendable (AudioChunk) -> Void,
        onConfirmation: @escaping @Sendable (AudioChunk) -> Void
    ) throws {
        guard !_isActive else { return }
        
        _isActive = true
        startTime = Date()
        processor = StreamingAudioProcessor()
        
        // Dual mode with separate callbacks - now receives sample positions
        try processor?.startStreaming(
            onHypothesis: { [weak self] audioData, startSample in
                self?.convertAndEmit(audioData: audioData, startSample: startSample, onChunk: onHypothesis)
            },
            onConfirmation: { [weak self] audioData, startSample in
                self?.convertAndEmit(audioData: audioData, startSample: startSample, onChunk: onConfirmation)
            }
        )
    }
    
    private func convertAndEmit(
        audioData: Data,
        startSample: Int,
        onChunk: @escaping @Sendable (AudioChunk) -> Void
    ) {
        // Extract samples from WAV data
        do {
            let samples = try AudioPreprocessor.extractPCMSamples(from: audioData)
            let startTime = Double(startSample) / Double(self.sampleRate)
            let duration = Double(samples.count) / Double(self.sampleRate)
            
            // Create AudioChunk with precise sample-based timing
            let chunk = AudioChunk(
                samples: samples,
                startTime: startTime,
                endTime: startTime + duration,
                overlapWithPrevious: 0,
                overlapWithNext: 0,
                sampleRate: self.sampleRate
            )
            
            onChunk(chunk)
        } catch {
            print("LiveAudioSource: ⚠️ Failed to process chunk: \(error)")
        }
    }
    
    public func stop() {
        guard _isActive else { return }
        
        _isActive = false
        processor?.stopStreaming()
        processor = nil
        startTime = nil
    }
}
