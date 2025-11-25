//
//  VoiceActivityDetector.swift
//  AudioServices
//
//  Simple energy-based Voice Activity Detection (VAD)
//  Detects speech vs silence using signal energy and zero-crossing rate
//

import Foundation

/// Simple energy-based VAD for speech detection
public class VoiceActivityDetector {
    
    // MARK: - Configuration
    
    /// Energy threshold (relative to recent average)
    private let energyThresholdRatio: Float = 3.5  // Increased from 2.0 - less sensitive
    
    /// Minimum energy floor (absolute)
    private let minEnergyThreshold: Float = 0.005  // Increased from 0.001 - require higher minimum
    
    /// Zero-crossing rate threshold (speech typically has lower ZCR than noise)
    private let maxZeroCrossingRate: Float = 0.4  // Decreased from 0.5 - stricter
    
    /// Minimum speech duration to consider valid (seconds)
    private let minSpeechDuration: TimeInterval = 0.5  // Increased from 0.3
    
    /// Window size for energy averaging (samples)
    private let windowSize: Int = 1600  // 100ms at 16kHz
    
    // MARK: - State
    
    /// Rolling average of background energy
    private var backgroundEnergy: Float = 0.0
    
    /// Adaptation rate for background energy (slower = more stable)
    private let adaptationRate: Float = 0.05  // Reduced from 0.1 - adapts more slowly
    
    // MARK: - Public Methods
    
    public init() {}
    
    /// Detect if audio chunk contains speech
    /// - Parameter samples: Audio samples to analyze
    /// - Returns: True if speech detected, false otherwise
    public func isSpeech(_ samples: [Float]) -> Bool {
        guard !samples.isEmpty else { return false }
        
        // Calculate signal energy (RMS)
        let energy = calculateRMS(samples)
        
        // Calculate zero-crossing rate
        let zcr = calculateZeroCrossingRate(samples)
        
        // Update background energy estimate (adapt slowly)
        if backgroundEnergy == 0 {
            backgroundEnergy = energy
        } else {
            backgroundEnergy = (1 - adaptationRate) * backgroundEnergy + adaptationRate * energy
        }
        
        // Speech detection criteria:
        // 1. Energy significantly above background
        // 2. Energy above minimum threshold
        // 3. Zero-crossing rate indicates speech (not white noise)
        let energyThreshold = max(backgroundEnergy * energyThresholdRatio, minEnergyThreshold)
        let hasEnergy = energy > energyThreshold
        let hasSpeechPattern = zcr < maxZeroCrossingRate
        
        return hasEnergy && hasSpeechPattern
    }
    
    /// Detect speech with confidence score
    /// - Parameter samples: Audio samples to analyze
    /// - Returns: Tuple of (isSpeech: Bool, confidence: Float)
    public func detectWithConfidence(_ samples: [Float]) -> (isSpeech: Bool, confidence: Float) {
        guard !samples.isEmpty else { return (false, 0.0) }
        
        let energy = calculateRMS(samples)
        let zcr = calculateZeroCrossingRate(samples)
        
        if backgroundEnergy == 0 {
            backgroundEnergy = energy
        } else {
            backgroundEnergy = (1 - adaptationRate) * backgroundEnergy + adaptationRate * energy
        }
        
        let energyThreshold = max(backgroundEnergy * energyThresholdRatio, minEnergyThreshold)
        
        // Calculate confidence based on energy ratio
        let energyRatio = energy / max(energyThreshold, 0.001)
        let energyConfidence = min(energyRatio, 1.0)
        
        // ZCR confidence (lower is better for speech)
        let zcrConfidence = 1.0 - min(zcr / maxZeroCrossingRate, 1.0)
        
        // Combined confidence
        let confidence = (energyConfidence + zcrConfidence) / 2.0
        
        let isSpeech = confidence > 0.5
        
        return (isSpeech, confidence)
    }
    
    /// Reset VAD state (e.g., when switching speakers or long pause)
    public func reset() {
        backgroundEnergy = 0.0
    }
    
    /// Find speech boundaries (pauses) in audio buffer
    /// Returns sample indices where speech pauses occur (good chunk boundaries)
    /// - Parameters:
    ///   - samples: Audio samples to analyze
    ///   - sampleRate: Sample rate in Hz
    ///   - minPauseDuration: Minimum pause duration in seconds
    /// - Returns: Array of sample indices where pauses occur
    public func findSpeechBoundaries(
        _ samples: [Float],
        sampleRate: Double = 16000.0,
        minPauseDuration: TimeInterval = 0.5
    ) -> [Int] {
        guard !samples.isEmpty else { return [] }
        
        var boundaries: [Int] = []
        let chunkSize = Int(sampleRate * 0.1)  // 100ms windows
        let minPauseSamples = Int(sampleRate * minPauseDuration)
        
        var pauseStartIndex: Int? = nil
        
        // Scan through audio in small windows
        var i = 0
        while i < samples.count {
            let endIdx = min(i + chunkSize, samples.count)
            let window = Array(samples[i..<endIdx])
            
            let (hasSpeech, _) = detectWithConfidence(window)
            
            if !hasSpeech {
                // Start of pause
                if pauseStartIndex == nil {
                    pauseStartIndex = i
                }
            } else {
                // Speech resumed - check if pause was long enough
                if let pauseStart = pauseStartIndex {
                    let pauseDuration = i - pauseStart
                    if pauseDuration >= minPauseSamples {
                        // Found a good boundary (middle of the pause)
                        let boundaryIndex = pauseStart + pauseDuration / 2
                        boundaries.append(boundaryIndex)
                    }
                    pauseStartIndex = nil
                }
            }
            
            i += chunkSize
        }
        
        return boundaries
    }
    
    /// Find best chunk boundary near target position
    /// Looks for speech pause within search window around target
    /// - Parameters:
    ///   - samples: Audio buffer
    ///   - targetPosition: Desired chunk boundary position
    ///   - searchWindow: Search window in samples (looks ±window around target)
    /// - Returns: Optimal boundary position, or target if no pause found
    public func findBestBoundaryNear(
        _ samples: [Float],
        targetPosition: Int,
        searchWindow: Int
    ) -> Int {
        guard targetPosition > 0 && targetPosition < samples.count else {
            return targetPosition
        }
        
        let windowStart = max(0, targetPosition - searchWindow)
        let windowEnd = min(samples.count, targetPosition + searchWindow)
        
        guard windowEnd > windowStart else { return targetPosition }
        
        let searchRegion = Array(samples[windowStart..<windowEnd])
        let chunkSize = Int(windowSize / 2)  // 50ms windows for fine-grained search
        
        var lowestEnergy: Float = .infinity
        var bestBoundary = targetPosition
        
        // Find the quietest point in the search window
        var i = 0
        while i < searchRegion.count - chunkSize {
            let window = Array(searchRegion[i..<(i + chunkSize)])
            let energy = calculateRMS(window)
            
            if energy < lowestEnergy {
                lowestEnergy = energy
                bestBoundary = windowStart + i + chunkSize / 2
            }
            
            i += chunkSize / 2  // 50% overlap
        }
        
        return bestBoundary
    }
    
    // MARK: - Segment Detection
    
    /// Speech segment with begin/end boundaries
    public struct SpeechSegment {
        public let beginSample: Int
        public let endSample: Int
        public let duration: TimeInterval
        public let avgConfidence: Float
        public let isComplete: Bool  // True if ended by silence, false if ongoing at buffer end
    }
    
    /// Detect complete speech segments (Begin → End)
    /// Returns segments where speech starts and ends with silence
    /// - Parameters:
    ///   - samples: Audio samples to analyze
    ///   - sampleRate: Sample rate in Hz
    ///   - minSilenceDuration: Minimum silence to mark segment end (seconds)
    ///   - minSegmentDuration: Minimum segment duration to return (seconds)
    /// - Returns: Array of detected speech segments
    public func detectSpeechSegments(
        _ samples: [Float],
        sampleRate: Double = 16000.0,
        minSilenceDuration: TimeInterval = 0.3,
        minSegmentDuration: TimeInterval = 0.5
    ) -> [SpeechSegment] {
        guard !samples.isEmpty else { return [] }
        
        var segments: [SpeechSegment] = []
        let chunkSize = Int(sampleRate * 0.05)  // 50ms windows
        let minSilenceSamples = Int(sampleRate * minSilenceDuration)
        let minSegmentSamples = Int(sampleRate * minSegmentDuration)
        
        var currentSegmentStart: Int? = nil
        var silenceStartIndex: Int? = nil
        var confidenceSum: Float = 0.0
        var confidenceCount: Int = 0
        
        // Scan through audio in small windows
        var i = 0
        while i < samples.count {
            let endIdx = min(i + chunkSize, samples.count)
            let window = Array(samples[i..<endIdx])
            
            let (hasSpeech, confidence) = detectWithConfidence(window)
            
            if hasSpeech {
                // Speech detected
                if currentSegmentStart == nil {
                    // Begin new segment (B)
                    currentSegmentStart = i
                    confidenceSum = 0.0
                    confidenceCount = 0
                }
                confidenceSum += confidence
                confidenceCount += 1
                silenceStartIndex = nil  // Reset silence counter
            } else {
                // Silence detected
                if currentSegmentStart != nil {
                    // We're in a segment, start counting silence
                    if silenceStartIndex == nil {
                        silenceStartIndex = i
                    } else {
                        // Check if silence is long enough to end segment
                        let silenceDuration = i - silenceStartIndex!
                        if silenceDuration >= minSilenceSamples {
                            // End segment (E) at middle of silence gap for better accuracy
                            let segmentEnd = silenceStartIndex! + (silenceDuration / 2)
                            let segmentDuration = segmentEnd - currentSegmentStart!
                            
                            if segmentDuration >= minSegmentSamples {
                                let avgConfidence = confidenceCount > 0 ? confidenceSum / Float(confidenceCount) : 0.0
                                segments.append(SpeechSegment(
                                    beginSample: currentSegmentStart!,
                                    endSample: segmentEnd,
                                    duration: Double(segmentDuration) / sampleRate,
                                    avgConfidence: avgConfidence,
                                    isComplete: true  // Ended by silence
                                ))
                            }
                            
                            currentSegmentStart = nil
                            silenceStartIndex = nil
                        }
                    }
                }
            }
            
            i += chunkSize
        }
        
        // Handle ongoing segment at end of buffer
        if let segmentStart = currentSegmentStart {
            let segmentEnd = samples.count
            let segmentDuration = segmentEnd - segmentStart
            if segmentDuration >= minSegmentSamples {
                let avgConfidence = confidenceCount > 0 ? confidenceSum / Float(confidenceCount) : 0.0
                segments.append(SpeechSegment(
                    beginSample: segmentStart,
                    endSample: segmentEnd,
                    duration: Double(segmentDuration) / sampleRate,
                    avgConfidence: avgConfidence,
                    isComplete: false  // Ongoing, not ended by silence
                ))
            }
        }
        
        return segments
    }
    
    // MARK: - Private Helpers
    
    /// Calculate Root Mean Square (RMS) energy
    private func calculateRMS(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0.0 }
        
        let sumOfSquares = samples.reduce(0.0) { $0 + $1 * $1 }
        return sqrt(sumOfSquares / Float(samples.count))
    }
    
    /// Calculate Zero-Crossing Rate
    /// Speech typically has lower ZCR than pure noise
    private func calculateZeroCrossingRate(_ samples: [Float]) -> Float {
        guard samples.count > 1 else { return 0.0 }
        
        var crossings = 0
        for i in 1..<samples.count {
            if (samples[i] >= 0 && samples[i-1] < 0) || (samples[i] < 0 && samples[i-1] >= 0) {
                crossings += 1
            }
        }
        
        return Float(crossings) / Float(samples.count)
    }
}
