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
        
        // Convert StreamingAudioProcessor's Data chunks to AudioChunk
        try processor?.startStreaming { [weak self] audioData in
            guard let self = self, let startTime = self.startTime else { return }
            
            // Extract samples from WAV data
            do {
                let samples = try AudioPreprocessor.extractPCMSamples(from: audioData)
                let currentTime = Date().timeIntervalSince(startTime)
                let duration = Double(samples.count) / Double(self.sampleRate)
                
                // Create AudioChunk
                let chunk = AudioChunk(
                    samples: samples,
                    startTime: max(0, currentTime - duration),
                    endTime: currentTime,
                    overlapWithPrevious: 0,  // StreamingAudioProcessor handles overlap internally
                    overlapWithNext: 0,
                    sampleRate: self.sampleRate
                )
                
                onChunk(chunk)
            } catch {
                print("LiveAudioSource: ⚠️ Failed to process chunk: \(error)")
            }
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
