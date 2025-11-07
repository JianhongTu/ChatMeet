//
//  StreamingAudioProcessor.swift
//  MeetingAssistantCore
//
//  Handles real-time audio processing with sliding window strategy
//

import Foundation
import AVFoundation

/// Processes audio in real-time using a sliding window strategy with ring buffer
class StreamingAudioProcessor: NSObject, @unchecked Sendable {
    
    // Configuration
    private let chunkDuration: TimeInterval = 5.0  // 5 second chunks
    private let sampleRate: Double = 16000.0
    private let channelCount: Int = 1
    
    // Ring buffer for audio samples
    private var audioRingBuffer: [Float] = []
    private let maxBufferSize: Int  // Maximum samples in buffer
    private var bufferWritePosition: Int = 0
    private var totalSamplesWritten: Int = 0
    
    // Audio engine components
    private var audioEngine: AVAudioEngine?
    private var inputNode: AVAudioInputNode?
    
    // Callback for processed audio chunks
    private var onAudioChunk: (@Sendable (Data) -> Void)?
    
    // State
    private var isRecording = false
    private var processingQueue = DispatchQueue(label: "com.chatmeet.audioprocessing", qos: .userInitiated)
    
    override init() {
        // Calculate buffer size for 30 seconds (matching Whisper's input length)
        self.maxBufferSize = Int(30.0 * sampleRate)
        self.audioRingBuffer = [Float](repeating: 0, count: maxBufferSize)
        super.init()
    }
    
    /// Start streaming audio processing
    /// - Parameter onChunk: Callback invoked with each 5-second audio chunk
    public func startStreaming(onChunk: @escaping @Sendable (Data) -> Void) throws {
        guard !isRecording else { return }
        
        self.onAudioChunk = onChunk
        self.isRecording = true
        
        // Reset buffer
        audioRingBuffer = [Float](repeating: 0, count: maxBufferSize)
        bufferWritePosition = 0
        totalSamplesWritten = 0
        
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
    
    /// Process incoming audio buffer and write to ring buffer
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
        
        // Write samples to ring buffer
        writeToRingBuffer(samples)
        
        // Check if we should process a chunk
        let chunkSize = Int(chunkDuration * sampleRate)
        if totalSamplesWritten >= chunkSize && totalSamplesWritten % chunkSize < samples.count {
            processChunk()
        }
    }
    
    /// Write samples to ring buffer with wraparound
    private func writeToRingBuffer(_ samples: [Float]) {
        for sample in samples {
            audioRingBuffer[bufferWritePosition] = sample
            bufferWritePosition = (bufferWritePosition + 1) % maxBufferSize
            totalSamplesWritten += 1
        }
    }
    
    /// Process a chunk of audio and invoke callback
    private func processChunk() {
        // Extract the last 30 seconds from ring buffer for Whisper
        let chunkSamples = extractCurrentWindow()
        
        // Convert to WAV format
        guard let wavData = convertToWAV(samples: chunkSamples) else {
            return
        }
        
        // Invoke callback with audio chunk
        onAudioChunk?(wavData)
    }
    
    /// Extract current 30-second window from ring buffer
    private func extractCurrentWindow() -> [Float] {
        let windowSize = min(totalSamplesWritten, maxBufferSize)
        var samples = [Float](repeating: 0, count: windowSize)
        
        // Read from ring buffer accounting for wraparound
        if totalSamplesWritten < maxBufferSize {
            // Buffer hasn't wrapped yet, just copy from beginning
            samples = Array(audioRingBuffer.prefix(windowSize))
        } else {
            // Buffer has wrapped, need to reconstruct order
            let readPosition = bufferWritePosition
            for i in 0..<windowSize {
                let ringIndex = (readPosition + i) % maxBufferSize
                samples[i] = audioRingBuffer[ringIndex]
            }
        }
        
        return samples
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
    public func getBufferStatus() -> (samplesInBuffer: Int, totalProcessed: Int) {
        return (min(totalSamplesWritten, maxBufferSize), totalSamplesWritten)
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
