//
//  StreamingContextStrategy.swift
//  TranscriptionServices
//
//  Strategy for real-time streaming transcription with small chunks
//

import Foundation

/// Context window strategy optimized for real-time streaming
public class StreamingContextStrategy: ContextWindowStrategy {
    public let windowSize: TimeInterval
    public let overlapDuration: TimeInterval
    
    /// Initialize with window and overlap configuration
    /// - Parameters:
    ///   - windowSize: Size of each window in seconds (default: 5s)
    ///   - overlapDuration: Overlap between windows in seconds (default: 1s)
    public init(windowSize: TimeInterval = 5.0, overlapDuration: TimeInterval = 1.0) {
        self.windowSize = windowSize
        self.overlapDuration = overlapDuration
    }
    
    public func chunkAudio(_ audio: AudioData) -> [AudioChunk] {
        var chunks: [AudioChunk] = []
        
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
            
            // Move to next chunk with overlap
            startSample += strideSamples
            
            // If remaining audio is smaller than stride, make sure we get it
            if startSample < totalSamples && (totalSamples - startSample) < strideSamples {
                startSample = max(0, totalSamples - windowSamples)
            }
        }
        
        return chunks
    }
}
