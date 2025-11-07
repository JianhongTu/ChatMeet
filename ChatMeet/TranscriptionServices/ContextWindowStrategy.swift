//
//  ContextWindowStrategy.swift
//  TranscriptionServices
//
//  Strategies for chunking audio based on context windows
//

import Foundation

/// Protocol for audio chunking strategies
public protocol ContextWindowStrategy {
    /// Window size in seconds
    var windowSize: TimeInterval { get }
    
    /// Overlap between consecutive windows
    var overlapDuration: TimeInterval { get }
    
    /// Break audio into chunks based on strategy
    /// - Parameter audio: The audio data to chunk
    /// - Returns: Array of audio chunks with timing information
    func chunkAudio(_ audio: AudioData) -> [AudioChunk]
    
    /// Merge transcribed segments, handling overlaps
    /// - Parameter segments: Transcription segments from chunked audio
    /// - Returns: Final merged text
    func mergeTranscriptions(_ segments: [TranscriptionSegment]) -> String
}

/// Extension with default merge implementation
extension ContextWindowStrategy {
    public func mergeTranscriptions(_ segments: [TranscriptionSegment]) -> String {
        guard !segments.isEmpty else { return "" }
        
        // Sort segments by start time
        let sortedSegments = segments.sorted { $0.startTime < $1.startTime }
        
        var result = sortedSegments[0].text
        
        // Merge subsequent segments, skipping overlap
        for i in 1..<sortedSegments.count {
            let segment = sortedSegments[i]
            let previousSegment = sortedSegments[i - 1]
            
            // Calculate overlap ratio
            let overlapTime = previousSegment.endTime - segment.startTime
            let segmentDuration = segment.endTime - segment.startTime
            
            if overlapTime > 0 && segmentDuration > 0 {
                let overlapRatio = overlapTime / segmentDuration
                
                // Skip overlapping portion of words
                let words = segment.text.split(separator: " ").map(String.init)
                let skipWords = Int(Double(words.count) * overlapRatio)
                let remainingWords = Array(words.dropFirst(min(skipWords, words.count)))
                
                if !remainingWords.isEmpty {
                    result += " " + remainingWords.joined(separator: " ")
                }
            } else {
                // No overlap, just append
                if !result.isEmpty && !segment.text.isEmpty {
                    result += " "
                }
                result += segment.text
            }
        }
        
        return result
    }
}
