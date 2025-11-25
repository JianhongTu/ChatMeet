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
    
    // Dual-mode configuration - now VAD-driven instead of fixed windows
    private let maxSegmentDuration: TimeInterval = 30.0  // Maximum segment before forced split
    private let minSilenceDuration: TimeInterval = 0.7   // Increased from 0.3 - require longer silence to end segment
    private let minConfirmationDuration: TimeInterval = 10.0  // Accumulate segments until 10s before confirming
    private let maxPauseBetweenSegments: TimeInterval = 2.0   // Max pause to still consider same utterance
    private let hypothesisUpdateInterval: TimeInterval = 0.5  // Update hypothesis every 0.5s (was 2.0s)
    private let sampleRate: Double = 16000.0
    private let channelCount: Int = 1
    
    // Simple accumulation buffer
    private var sampleBuffer: [Float] = []
    private var bufferStartSample: Int = 0  // Absolute sample position of buffer[0]
    
    // VAD-driven state tracking
    private var currentSegmentStart: Int? = nil  // B (begin) position
    private var lastHypothesisUpdate: Int = 0     // Last hypothesis sent
    private var lastConfirmedPosition: Int = 0    // Last confirmed segment end
    private var accumulatedSegments: [(begin: Int, end: Int)] = []  // Segments waiting for confirmation
    
    // Audio engine components
    private var audioEngine: AVAudioEngine?
    private var inputNode: AVAudioInputNode?
    
    // Callbacks for processed audio chunks (now includes start sample position)
    private var onHypothesisChunk: (@Sendable (Data, Int) -> Void)?  // (wavData, startSample)
    private var onConfirmationChunk: (@Sendable (Data, Int) -> Void)?  // (wavData, startSample)
    
    // Voice Activity Detection
    private let vad = VoiceActivityDetector()
    private let useSpeechBoundaries: Bool = true  // Enable boundary detection
    private let boundarySearchWindow: Double = 1.0  // Search ±1s around target
    
    // State
    private var isRecording = false
    private var processingQueue = DispatchQueue(label: "com.chatmeet.audioprocessing", qos: .userInitiated)
    
    override init() {
        super.init()
    }
    
    /// Start streaming audio processing with dual-mode chunks
    /// - Parameters:
    ///   - onHypothesis: Callback for ongoing segments (wavData, startSample)
    ///   - onConfirmation: Callback for complete segments (wavData, startSample)
    public func startStreaming(
        onHypothesis: @escaping @Sendable (Data, Int) -> Void,
        onConfirmation: @escaping @Sendable (Data, Int) -> Void
    ) throws {
        guard !isRecording else { return }
        
        self.onHypothesisChunk = onHypothesis
        self.onConfirmationChunk = onConfirmation
        self.isRecording = true
        
        // Reset buffer and VAD
        sampleBuffer = []
        bufferStartSample = 0
        currentSegmentStart = nil
        lastHypothesisUpdate = 0
        lastConfirmedPosition = 0
        accumulatedSegments = []
        vad.reset()
        
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
    
    /// Process chunks using VAD-driven segmentation with accumulation
    /// Accumulates multiple short segments until enough context before confirming
    private func processChunks() {
        let totalSamplesCollected = bufferStartSample + sampleBuffer.count
        let hypothesisUpdateSamples = Int(hypothesisUpdateInterval * sampleRate)
        
        // Detect speech segments in current buffer
        let segments = vad.detectSpeechSegments(
            sampleBuffer,
            sampleRate: sampleRate,
            minSilenceDuration: minSilenceDuration,
            minSegmentDuration: 0.5
        )
        
        // Process new complete segments - add to accumulation
        var hasNewSegments = false
        for segment in segments {
            let absoluteBegin = bufferStartSample + segment.beginSample
            let absoluteEnd = bufferStartSample + segment.endSample
            
            // Skip if we've already confirmed past this segment
            if absoluteEnd <= lastConfirmedPosition {
                continue
            }
            
            if segment.isComplete {
                // Complete segment (B→E detected) - add to accumulation if not already added
                let alreadyAdded = accumulatedSegments.contains { $0.begin == absoluteBegin && $0.end == absoluteEnd }
                if !alreadyAdded {
                    accumulatedSegments.append((begin: absoluteBegin, end: absoluteEnd))
                    hasNewSegments = true
                    print("➕ Added segment to accumulation: \(accumulatedSegments.count) segments, latest ended at \(String(format: "%.2f", Double(absoluteEnd) / sampleRate))s")
                }
            }
        }
        
        // Check if we should confirm accumulated segments
        // Only check when we have new segments or periodically
        if !accumulatedSegments.isEmpty && hasNewSegments,
           let firstAccumulated = accumulatedSegments.first,
           let lastAccumulated = accumulatedSegments.last {
            
            let accumulatedDuration = Double(lastAccumulated.end - firstAccumulated.begin) / sampleRate
            let timeSinceLastSegment = Double(totalSamplesCollected - lastAccumulated.end) / sampleRate
            
            // Confirm if:
            // 1. Accumulated enough duration (10s), OR
            // 2. Long pause after last segment (2s of silence), OR
            // 3. Too many segments accumulated (safety)
            let hasEnoughDuration = accumulatedDuration >= minConfirmationDuration
            let hasLongPause = timeSinceLastSegment >= maxPauseBetweenSegments
            let tooManySegments = accumulatedSegments.count >= 5
            
            let shouldConfirm = hasEnoughDuration || hasLongPause || tooManySegments
            
            print("🔍 Accumulation check: duration=\(String(format: "%.1f", accumulatedDuration))s, pauseSince=\(String(format: "%.2f", timeSinceLastSegment))s, segments=\(accumulatedSegments.count)")
            
            if shouldConfirm {
                let reason = hasEnoughDuration ? "duration" : (hasLongPause ? "long pause" : "too many segments")
                print("✅ CONFIRMATION: Sending \(accumulatedSegments.count) segments [\(String(format: "%.1f", accumulatedDuration))s, reason=\(reason)]")
                
                // Send accumulated segments as one confirmation chunk
                let startInBuffer = firstAccumulated.begin - bufferStartSample
                let endInBuffer = lastAccumulated.end - bufferStartSample
                
                if startInBuffer >= 0 && endInBuffer <= sampleBuffer.count {
                    let chunk = Array(sampleBuffer[startInBuffer..<endInBuffer])
                    let absoluteStart = firstAccumulated.begin
                    let chunkDuration = Double(chunk.count) / sampleRate
                    
                    print("📤 Sending confirmation chunk: \(chunk.count) samples (\(String(format: "%.2f", chunkDuration))s)")
                    
                    if let wavData = convertToWAV(samples: chunk) {
                        onConfirmationChunk?(wavData, absoluteStart)
                    }
                    
                    // Update positions and clear accumulation
                    lastConfirmedPosition = lastAccumulated.end
                    lastHypothesisUpdate = lastAccumulated.end
                    accumulatedSegments = []
                    currentSegmentStart = nil
                }
            }
        }
        
        // Check for ongoing segments to send as HYPOTHESIS
        for segment in segments {
            let absoluteBegin = bufferStartSample + segment.beginSample
            let absoluteEnd = bufferStartSample + segment.endSample
            
            if absoluteEnd <= lastConfirmedPosition {
                continue
            }
            
            if !segment.isComplete {
                // Ongoing segment (B detected, no E yet) - send as HYPOTHESIS
                // Include accumulated segments + current ongoing segment
                if totalSamplesCollected - lastHypothesisUpdate >= hypothesisUpdateSamples {
                    let hypothesisStart: Int
                    if let firstAccumulated = accumulatedSegments.first {
                        hypothesisStart = firstAccumulated.begin
                    } else {
                        hypothesisStart = absoluteBegin
                    }
                    
                    let startInBuffer = hypothesisStart - bufferStartSample
                    let endInBuffer = segment.endSample
                    
                    if startInBuffer >= 0 && endInBuffer <= sampleBuffer.count {
                        let chunk = Array(sampleBuffer[startInBuffer..<endInBuffer])
                        let absoluteStart = hypothesisStart
                        let duration = Double(absoluteEnd - hypothesisStart) / sampleRate
                        
                        let (hasSpeech, confidence) = vad.detectWithConfidence(chunk)
                        if hasSpeech {
                            if let wavData = convertToWAV(samples: chunk) {
                                onHypothesisChunk?(wavData, absoluteStart)
                            }
                            
                            let segCount = accumulatedSegments.count + 1
                            print("💭 HYPOTHESIS: \(segCount) segment(s) [\(String(format: "%.1f", duration))s, conf=\(String(format: "%.2f", confidence))]")
                            
                            lastHypothesisUpdate = totalSamplesCollected
                            if currentSegmentStart == nil {
                                currentSegmentStart = hypothesisStart
                            }
                        }
                    }
                }
            }
        }
        
        // Handle very long accumulated duration (force confirmation)
        if let firstAccumulated = accumulatedSegments.first {
            let accumulatedDuration = Double(totalSamplesCollected - firstAccumulated.begin) / sampleRate
            if accumulatedDuration > maxSegmentDuration {
                print("⚠️ Accumulated duration too long (\(String(format: "%.1f", accumulatedDuration))s), forcing confirmation")
                
                guard let lastAccumulated = accumulatedSegments.last else { return }
                let startInBuffer = firstAccumulated.begin - bufferStartSample
                let endInBuffer = lastAccumulated.end - bufferStartSample
                
                if startInBuffer >= 0 && endInBuffer <= sampleBuffer.count {
                    let chunk = Array(sampleBuffer[startInBuffer..<endInBuffer])
                    
                    if let wavData = convertToWAV(samples: chunk) {
                        onConfirmationChunk?(wavData, firstAccumulated.begin)
                    }
                    
                    lastConfirmedPosition = lastAccumulated.end
                    lastHypothesisUpdate = lastAccumulated.end
                    accumulatedSegments = []
                    currentSegmentStart = nil
                }
            }
        }
        
        // Trim old samples from buffer
        let oldestNeededPosition = lastConfirmedPosition
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
