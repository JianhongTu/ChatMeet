//
//  SummaryModels.swift
//  MeetingAssistantCore
//
//  Data models for agentic summary management
//

import Foundation

/// A single bullet point in the meeting summary
public struct SummaryBulletPoint: Identifiable, Codable, Equatable {
    public let id: Int
    public var content: String
    public let createdAt: Date
    public var updatedAt: Date
    
    public init(id: Int, content: String, createdAt: Date = Date(), updatedAt: Date = Date()) {
        self.id = id
        self.content = content
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }
}

/// Actions that the LLM can perform on the summary
public enum SummaryAction: Equatable {
    case append(content: String)
    case update(id: Int, content: String)
    case delete(id: Int)
    case noAction
    
    /// Parse actions from LLM summary output
    /// Only matches lines starting with "* " (markdown bullet style)
    static func parse(from text: String) -> [SummaryAction] {
        var actions: [SummaryAction] = []
        
        // Clean up the text
        let cleaned = text.trimmingCharacters(in: .whitespacesAndNewlines)
        
        if cleaned.isEmpty {
            print("⚠️ SummaryAction: Empty output")
            return [.noAction]
        }
        
        // Look for markdown bullet points starting with "* "
        let lines = cleaned.split(separator: "\n", omittingEmptySubsequences: true)
        
        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            
            // Only match lines starting with "* "
            if trimmed.hasPrefix("• ") {
                let content = String(trimmed.dropFirst(2)).trimmingCharacters(in: .whitespaces)
                
                if !content.isEmpty && content.count > 3 {
                    actions.append(.append(content: content))
                    print("✅ SummaryAction: Parsed bullet: \(content)")
                }
            }
        }
        
        if actions.isEmpty {
            print("⚠️ SummaryAction: No markdown bullets found (lines must start with '* ')")
        } else {
            print("✅ SummaryAction: Parsed \(actions.count) bullet points")
        }
        
        return actions.isEmpty ? [.noAction] : actions
    }
}
