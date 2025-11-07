//
//  AudioData.swift
//  AudioServices
//
//  Unified audio data structure across the application
//

import Foundation

/// Represents audio data in a standardized format
public struct AudioData {
    /// Audio samples as normalized float values (-1.0 to 1.0)
    public let samples: [Float]
    
    /// Sample rate in Hz (e.g., 16000)
    public let sampleRate: Int
    
    /// Duration of the audio in seconds
    public var duration: TimeInterval {
        return Double(samples.count) / Double(sampleRate)
    }
    
    /// Original audio format
    public let format: AudioFormat
    
    /// Optional metadata
    public let metadata: AudioMetadata?
    
    public init(samples: [Float], sampleRate: Int, format: AudioFormat, metadata: AudioMetadata? = nil) {
        self.samples = samples
        self.sampleRate = sampleRate
        self.format = format
        self.metadata = metadata
    }
}

/// Audio format types
public enum AudioFormat {
    case wav
    case m4a
    case sphere  // NIST SPHERE format
    case raw
}

/// Metadata about the audio
public struct AudioMetadata {
    public let channelCount: Int
    public let bitDepth: Int?
    public let sourceURL: URL?
    
    public init(channelCount: Int, bitDepth: Int? = nil, sourceURL: URL? = nil) {
        self.channelCount = channelCount
        self.bitDepth = bitDepth
        self.sourceURL = sourceURL
    }
}

/// Represents a chunk of audio with timing information
public struct AudioChunk {
    /// Audio samples for this chunk
    public let samples: [Float]
    
    /// Start time relative to the original audio (in seconds)
    public let startTime: TimeInterval
    
    /// End time relative to the original audio (in seconds)
    public let endTime: TimeInterval
    
    /// Duration that overlaps with the previous chunk
    public let overlapWithPrevious: TimeInterval
    
    /// Duration that overlaps with the next chunk
    public let overlapWithNext: TimeInterval
    
    /// Sample rate
    public let sampleRate: Int
    
    /// Computed duration
    public var duration: TimeInterval {
        return endTime - startTime
    }
    
    public init(
        samples: [Float],
        startTime: TimeInterval,
        endTime: TimeInterval,
        overlapWithPrevious: TimeInterval,
        overlapWithNext: TimeInterval,
        sampleRate: Int
    ) {
        self.samples = samples
        self.startTime = startTime
        self.endTime = endTime
        self.overlapWithPrevious = overlapWithPrevious
        self.overlapWithNext = overlapWithNext
        self.sampleRate = sampleRate
    }
}
