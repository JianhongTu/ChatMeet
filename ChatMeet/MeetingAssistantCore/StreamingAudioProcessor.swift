//
//  StreamingAudioProcessor.swift
//  MeetingAssistantCore
//
//  Handles real-time audio processing with sliding window strategy
//

import Foundation
import AVFoundation

/// Processes audio in real-time using FluidAudio strategy
/// 15s chunks with 2s overlap, stateless processing
class StreamingAudioProcessor: NSObject, @unchecked Sendable {
    
    // Dual-mode configuration for low latency
    private let hypothesisChunkDuration: TimeInterval = 5.0   // 5s chunks for reasonable latency
    private let confirmationChunkDuration: TimeInterval = 15.0 // 15s accurate chunks for final text
    private let overlapDuration: TimeInterval = 1.0  // 1s overlap for hypothesis
    private let sampleRate: Double = 16000.0
    private let channelCount: Int = 1
    
    // Simple accumulation buffer (not ring buffer)
    private var sampleBuffer: [Float] = []
    private var bufferStartSample: Int = 0  // Absolute sample position of buffer[0]
    private var lastHypothesisPosition: Int = 0  // Track where last hypothesis chunk was sent
    private var lastConfirmationPosition: Int = 0  // Track where last confirmation chunk was sent
    
    // Audio engine components
    private var audioEngine: AVAudioEngine?
    private var inputNode: AVAudioInputNode?
    
    // Callbacks for processed audio chunks
    private var onHypothesisChunk: (@Sendable (Data) -> Void)?
    private var onConfirmationChunk: (@Sendable (Data) -> Void)?
    
    // State
    private var isRecording = false
    private var processingQueue = DispatchQueue(label: "com.chatmeet.audioprocessing", qos: .userInitiated)
    
    override init() {
        super.init()
    }
    
    /// Start streaming audio processing with dual-mode chunks
    /// - Parameters:
    ///   - onHypothesis: Callback for fast 3s chunks (low latency preview)
    ///   - onConfirmation: Callback for accurate 15s chunks (final transcription)
    public func startStreaming(
        onHypothesis: @escaping @Sendable (Data) -> Void,
        onConfirmation: @escaping @Sendable (Data) -> Void
    ) throws {
        guard !isRecording else { return }
        
        self.onHypothesisChunk = onHypothesis
        self.onConfirmationChunk = onConfirmation
        self.isRecording = true
        
        // Reset buffer
        sampleBuffer = []
        bufferStartSample = 0
        lastHypothesisPosition = 0
        lastConfirmationPosition = 0
        
        // Setup audio engine
        try setupAudioEngine()
    }
    
    /// Stop streaming audio processing
    public func stopStreaming() {
        guard isRecording else { return }
        
        isRecording = false
        audioEngine?.stop()
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine = nil
        inputNode = nil
    }
    
    /// Setup AVAudioEngine for real-time audio capture
    private func setupAudioEngine() throws {
        audioEngine = AVAudioEngine()
        guard let engine = audioEngine else {
            throw StreamingError.audioEngineSetupFailed
        }
        
        inputNode = engine.inputNode
        guard let input = inputNode else {
            throw StreamingError.audioEngineSetupFailed
        }
        
        // Get the input format (typically 48kHz on macOS)
        let inputFormat = input.inputFormat(forBus: 0)
        
        // Create our desired format (16kHz mono)
        guard let desiredFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: UInt32(channelCount),
            interleaved: false
        ) else {
            throw StreamingError.audioFormatError
        }
        
        // Install tap to capture audio
        input.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { [weak self] buffer, time in
            guard let self = self else { return }
            self.processingQueue.async {
                self.processAudioBuffer(buffer, inputFormat: inputFormat, outputFormat: desiredFormat)
            }
        }
        
        // Start the engine
        try engine.start()
    }
    
    /// Process incoming audio buffer and accumulate samples
    private func processAudioBuffer(_ buffer: AVAudioPCMBuffer, inputFormat: AVAudioFormat, outputFormat: AVAudioFormat) {
        guard isRecording else { return }
        
        // Convert format if needed (resample from input rate to 16kHz)
        guard let convertedBuffer = convertAudioFormat(buffer, from: inputFormat, to: outputFormat) else {
            return
        }
        
        // Extract float samples
        guard let channelData = convertedBuffer.floatChannelData else { return }
        let frameLength = Int(convertedBuffer.frameLength)
        let samples = Array(UnsafeBufferPointer(start: channelData[0], count: frameLength))
        
        // Accumulate samples in buffer
        sampleBuffer.append(contentsOf: samples)
        
        // Process chunks when we have enough
        processChunks()
    }
    
    /// Process chunks in dual mode: fast hypothesis + accurate confirmation
    /// Both modes read from the same buffer without consuming it
    private func processChunks() {
        let hypothesisMinSize = Int(5.0 * sampleRate)  // 80,000 samples (5s minimum for first hypothesis)
        let confirmationSize = Int(confirmationChunkDuration * sampleRate)  // 240,000 samples (15s)
        let hypothesisStride = Int(4.0 * sampleRate)  // 64,000 samples (4s stride for update frequency)
        let confirmationStride = Int(13.0 * sampleRate)  // 208,000 samples (13s stride)
        
        let totalSamplesCollected = bufferStartSample + sampleBuffer.count
        
        // Process confirmation chunks (15s) first - this is the authoritative base
        while totalSamplesCollected - lastConfirmationPosition >= confirmationSize {
            let startInBuffer = lastConfirmationPosition - bufferStartSample
            
            if startInBuffer >= 0 && startInBuffer + confirmationSize <= sampleBuffer.count {
                let chunk = Array(sampleBuffer[startInBuffer..<(startInBuffer + confirmationSize)])
                
                if let wavData = convertToWAV(samples: chunk) {
                    onConfirmationChunk?(wavData)
                }
                
                lastConfirmationPosition += confirmationStride
                
                // Reset hypothesis position to read AFTER this confirmation chunk
                // This prevents hypothesis from reading already-confirmed audio
                if lastHypothesisPosition < lastConfirmationPosition {
                    lastHypothesisPosition = lastConfirmationPosition
                }
            } else {
                break
            }
        }
        
        // Process hypothesis chunks - CUMULATIVE from confirmation position
        // Hypothesis shows growing preview of all audio after confirmation
        let hypothesisStart = lastConfirmationPosition
        let availableSamples = totalSamplesCollected - hypothesisStart
        
        // Only emit hypothesis if we have enough samples AND it's been at least 4s since last update
        if availableSamples >= hypothesisMinSize && 
           totalSamplesCollected - lastHypothesisPosition >= hypothesisStride {
            
            let startInBuffer = hypothesisStart - bufferStartSample
            let endInBuffer = startInBuffer + availableSamples
            
            if startInBuffer >= 0 && endInBuffer <= sampleBuffer.count {
                // Send ALL samples from confirmation position to current position (cumulative)
                let chunk = Array(sampleBuffer[startInBuffer..<endInBuffer])
                
                if let wavData = convertToWAV(samples: chunk) {
                    onHypothesisChunk?(wavData)
                }
                
                lastHypothesisPosition = totalSamplesCollected
            }
        }
        
        // Trim old samples from buffer to prevent unbounded growth
        // Only trim samples that BOTH modes are done with
        let oldestNeededPosition = min(lastConfirmationPosition, lastConfirmationPosition)
        let samplesToTrim = oldestNeededPosition - bufferStartSample
        
        if samplesToTrim > 0 && samplesToTrim < sampleBuffer.count {
            sampleBuffer.removeFirst(samplesToTrim)
            bufferStartSample += samplesToTrim
        }
    }
    
    /// Convert audio format using AVAudioConverter
    private func convertAudioFormat(_ buffer: AVAudioPCMBuffer, from inputFormat: AVAudioFormat, to outputFormat: AVAudioFormat) -> AVAudioPCMBuffer? {
        // If formats match, no conversion needed
        if inputFormat.sampleRate == outputFormat.sampleRate && 
           inputFormat.channelCount == outputFormat.channelCount {
            return buffer
        }
        
        guard let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
            return nil
        }
        
        // Calculate output frame capacity
        let inputFrameCount = buffer.frameLength
        let ratio = outputFormat.sampleRate / inputFormat.sampleRate
        let outputFrameCapacity = AVAudioFrameCount(Double(inputFrameCount) * ratio)
        
        guard let outputBuffer = AVAudioPCMBuffer(
            pcmFormat: outputFormat,
            frameCapacity: outputFrameCapacity
        ) else {
            return nil
        }
        
        var error: NSError?
        let inputBlock: AVAudioConverterInputBlock = { inNumPackets, outStatus in
            outStatus.pointee = .haveData
            return buffer
        }
        
        converter.convert(to: outputBuffer, error: &error, withInputFrom: inputBlock)
        
        if error != nil {
            return nil
        }
        
        return outputBuffer
    }
    
    /// Convert float samples to WAV format (wrapper around shared utility)
    private func convertToWAV(samples: [Float]) -> Data? {
        return AudioPreprocessor.convertSamplesToWAV(samples, sampleRate: Int(sampleRate))
    }
    
    /// Get current buffer status
    public func getBufferStatus() -> (samplesInBuffer: Int, absolutePosition: Int) {
        return (sampleBuffer.count, bufferStartSample)
    }
}

/// Errors that can occur during streaming
enum StreamingError: LocalizedError {
    case audioEngineSetupFailed
    case audioFormatError
    case processingError
    
    var errorDescription: String? {
        switch self {
        case .audioEngineSetupFailed:
            return "Failed to setup audio engine"
        case .audioFormatError:
            return "Audio format conversion error"
        case .processingError:
            return "Audio processing error"
        }
    }
}
