//
//  TranscriptionProtocols.swift
//  TranscriptionServices
//
//  Core protocols for the transcription service layer
//

import Foundation

/// Mode of transcription operation
public enum TranscriptionMode {
    /// Real-time streaming with small chunks
    case streaming(chunkSize: TimeInterval)
    
    /// Batch processing with large windows
    case batch(maxWindow: TimeInterval)
    
    /// Automatically decide based on audio length
    case auto
}

/// Progress information during transcription
public struct TranscriptionProgress {
    /// How much audio has been processed (in seconds)
    public let processedDuration: TimeInterval
    
    /// Total audio duration (in seconds)
    public let totalDuration: TimeInterval
    
    /// Partial transcription text accumulated so far
    public let partialText: String
    
    /// Average confidence score (0.0 to 1.0)
    public let confidence: Double?
    
    /// Percentage completed (0.0 to 1.0)
    public var percentComplete: Double {
        guard totalDuration > 0 else { return 0 }
        return min(processedDuration / totalDuration, 1.0)
    }
    
    public init(
        processedDuration: TimeInterval,
        totalDuration: TimeInterval,
        partialText: String,
        confidence: Double? = nil
    ) {
        self.processedDuration = processedDuration
        self.totalDuration = totalDuration
        self.partialText = partialText
        self.confidence = confidence
    }
}

/// A segment of transcribed text with timing
public struct TranscriptionSegment {
    /// The transcribed text
    public let text: String
    
    /// Start time in the audio (seconds)
    public let startTime: TimeInterval
    
    /// End time in the audio (seconds)
    public let endTime: TimeInterval
    
    /// Confidence score for this segment
    public let confidence: Double?
    
    public init(text: String, startTime: TimeInterval, endTime: TimeInterval, confidence: Double? = nil) {
        self.text = text
        self.startTime = startTime
        self.endTime = endTime
        self.confidence = confidence
    }
}

/// Metadata about the transcription process
public struct TranscriptionMetadata {
    /// Model used for transcription
    public let modelName: String
    
    /// Sample rate of the processed audio
    public let sampleRate: Int
    
    /// Duration of the audio
    public let audioDuration: TimeInterval
    
    /// Time taken to process
    public let processingTime: TimeInterval
    
    /// Mode used for transcription
    public let mode: TranscriptionMode
    
    public init(
        modelName: String,
        sampleRate: Int,
        audioDuration: TimeInterval,
        processingTime: TimeInterval,
        mode: TranscriptionMode
    ) {
        self.modelName = modelName
        self.sampleRate = sampleRate
        self.audioDuration = audioDuration
        self.processingTime = processingTime
        self.mode = mode
    }
}

/// Result of transcription
public struct TranscriptionResult {
    /// Complete transcribed text
    public let text: String
    
    /// Individual segments with timing
    public let segments: [TranscriptionSegment]
    
    /// Metadata about the transcription
    public let metadata: TranscriptionMetadata
    
    public init(text: String, segments: [TranscriptionSegment], metadata: TranscriptionMetadata) {
        self.text = text
        self.segments = segments
        self.metadata = metadata
    }
}
