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
    private var bulletPoints: [SummaryBulletPoint] = []
    private var nextID: Int = 1
    
    public init() {
        self.summarizationModel = SummarizationModel()
    }
    
    /// Agentic workflow: Update summary based on new transcript
    /// - Parameters:
    ///   - newText: New transcript text
    ///   - onProgress: Optional callback for action streaming
    /// - Returns: Updated list of bullet points
    public func updateSummary(with newText: String, onProgress: (@Sendable (String) -> Void)? = nil) async throws -> [SummaryBulletPoint] {
        // Validate input
        guard !newText.isEmpty else {
            throw SummaryError.emptyText
        }
        
        // Ask LLM to decide what actions to take
        let actions = try await summarizationModel.analyzeAndDecide(
            newText: newText,
            currentBulletPoints: bulletPoints,
            onProgress: onProgress
        )
        
        // Apply actions to bullet points
        for action in actions {
            applyAction(action)
        }
        
        return bulletPoints
    }
    
    /// Apply a single action to the bullet points list
    private func applyAction(_ action: SummaryAction) {
        switch action {
        case .append(let content):
            let newPoint = SummaryBulletPoint(id: nextID, content: content)
            bulletPoints.append(newPoint)
            nextID += 1
            print("✅ SummaryService: APPENDED bullet \(newPoint.id): \(content)")
            
        case .update(let id, let content):
            if let index = bulletPoints.firstIndex(where: { $0.id == id }) {
                bulletPoints[index].content = content
                bulletPoints[index].updatedAt = Date()
                print("✏️ SummaryService: UPDATED bullet \(id): \(content)")
            } else {
                print("⚠️ SummaryService: UPDATE failed - bullet \(id) not found")
            }
            
        case .delete(let id):
            if let index = bulletPoints.firstIndex(where: { $0.id == id }) {
                bulletPoints.remove(at: index)
                print("🗑️ SummaryService: DELETED bullet \(id)")
            } else {
                print("⚠️ SummaryService: DELETE failed - bullet \(id) not found")
            }
            
        case .noAction:
            print("⏸️ SummaryService: No action needed")
        }
    }
    
    /// Reset the summary state
    public func reset() {
        bulletPoints = []
        nextID = 1
    }
    
    /// Get current bullet points
    public func getCurrentBulletPoints() -> [SummaryBulletPoint] {
        return bulletPoints
    }
    
    /// Summarize text into bullet points (legacy method for batch mode)
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
