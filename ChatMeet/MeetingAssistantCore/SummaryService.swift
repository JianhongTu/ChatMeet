//
//  SummaryService.swift
//  MeetingAssistantCore
//
//  Service for summarizing text into bullet points
//

import Foundation

/// Service that handles text summarization using a language model
class SummaryService: @unchecked Sendable {
    
    private let summarizationModel: SummarizationModel
    
    public init() {
        self.summarizationModel = SummarizationModel()
    }
    
    /// Summarize text into bullet points
    /// - Parameters:
    ///   - text: Text to summarize
    ///   - onProgress: Optional callback for real-time streaming
    /// - Returns: Bullet point summary
    public func summarize(_ text: String, onProgress: (@Sendable (String) -> Void)? = nil) async throws -> String {
        // Validate input text
        guard !text.isEmpty else {
            throw SummaryError.emptyText
        }
        
        // Process through summarization model with streaming
        let summary = try await summarizationModel.summarize(text, onProgress: onProgress)
        
        return summary
    }
}

/// Errors that can occur during summarization
public enum SummaryError: LocalizedError {
    case emptyText
    case modelNotLoaded
    case summarizationFailed(String)
    
    public var errorDescription: String? {
        switch self {
        case .emptyText:
            return "No text to summarize"
        case .modelNotLoaded:
            return "Summarization model is not loaded"
        case .summarizationFailed(let reason):
            return "Summarization failed: \(reason)"
        }
    }
}
