//
//  BatchContextStrategy.swift
//  TranscriptionServices
//
//  Strategy for batch transcription with large windows
//

import Foundation

/// Context window strategy optimized for batch processing
public class BatchContextStrategy: ContextWindowStrategy {
    public let windowSize: TimeInterval
    public let overlapDuration: TimeInterval
    
    /// Initialize with window and overlap configuration
    /// - Parameters:
    ///   - windowSize: Size of each window in seconds (default: 20s)
    ///   - overlapDuration: Overlap between windows in seconds (default: 2s)
    public init(windowSize: TimeInterval = 20.0, overlapDuration: TimeInterval = 2.0) {
        self.windowSize = windowSize
        self.overlapDuration = overlapDuration
    }
    
    public func chunkAudio(_ audio: AudioData) -> [AudioChunk] {
        var chunks: [AudioChunk] = []
        
        // If audio is shorter than window size, return as single chunk
        if audio.duration <= windowSize {
            let chunk = AudioChunk(
                samples: audio.samples,
                startTime: 0,
                endTime: audio.duration,
                overlapWithPrevious: 0,
                overlapWithNext: 0,
                sampleRate: audio.sampleRate
            )
            return [chunk]
        }
        
        let totalSamples = audio.samples.count
        let windowSamples = Int(windowSize * Double(audio.sampleRate))
        let overlapSamples = Int(overlapDuration * Double(audio.sampleRate))
        let strideSamples = windowSamples - overlapSamples
        
        var startSample = 0
        
        while startSample < totalSamples {
            let endSample = min(startSample + windowSamples, totalSamples)
            let chunkSamples = Array(audio.samples[startSample..<endSample])
            
            // Determine overlaps
            let hasOverlapBefore = startSample > 0
            let hasOverlapAfter = endSample < totalSamples
            
            let chunk = AudioChunk(
                samples: chunkSamples,
                startTime: Double(startSample) / Double(audio.sampleRate),
                endTime: Double(endSample) / Double(audio.sampleRate),
                overlapWithPrevious: hasOverlapBefore ? overlapDuration : 0,
                overlapWithNext: hasOverlapAfter ? overlapDuration : 0,
                sampleRate: audio.sampleRate
            )
            
            chunks.append(chunk)
            
            // Move to next chunk
            if endSample >= totalSamples {
                break
            }
            
            startSample += strideSamples
        }
        
        return chunks
    }
}
