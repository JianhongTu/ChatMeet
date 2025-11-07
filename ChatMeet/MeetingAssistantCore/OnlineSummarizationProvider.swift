//
//  OnlineSummarizationProvider.swift
//  MeetingAssistantCore
//
//  Online API provider for text summarization (OpenAI/compatible APIs)
//

import Foundation

/// Provider for online summarization APIs (OpenAI, Anthropic, etc.)
class OnlineSummarizationProvider: @unchecked Sendable {
    
    private let apiKey: String
    private let apiEndpoint: String
    private let model: String
    
    /// Initialize with API configuration
    /// - Parameters:
    ///   - apiKey: API key for authentication
    ///   - apiEndpoint: API endpoint base URL (defaults to Nautilus)
    ///   - model: Model name to use (defaults to gemma3)
    public init(apiKey: String, apiEndpoint: String = "https://ellm.nrp-nautilus.io/v1/chat/completions", model: String = "gemma3") {
        self.apiKey = apiKey
        self.apiEndpoint = apiEndpoint
        self.model = model
    }
    
    /// Summarize text using online API
    /// - Parameters:
    ///   - text: Text to summarize
    ///   - onProgress: Optional callback for streaming tokens (if supported)
    /// - Returns: Summary text
    public func summarize(_ text: String, onProgress: (@Sendable (String) -> Void)? = nil) async throws -> String {
        // Create the request
        guard let url = URL(string: apiEndpoint) else {
            throw OnlineSummarizationError.invalidEndpoint
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        // Create the messages
        let systemMessage = """
        You are a helpful meeting assistant. Summarize the key points from meeting transcripts. \
        Format your response as bullet points using markdown style with • as the bullet character.
        """
        
        let userMessage = """
        Create a high-level summary of the provided transcript with one item on each line.
        
        \(text)
        """
        
        // Build request body
        let requestBody: [String: Any] = [
            "model": model,
            "messages": [
                ["role": "system", "content": systemMessage],
                ["role": "user", "content": userMessage]
            ],
            "temperature": 0.7,
            "max_tokens": 500
        ]
        
        request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        
        // Debug logging
        print("🌐 OnlineSummarizationProvider: Making request to \(apiEndpoint)")
        print("🌐 Model: \(model)")
        print("🌐 API Key (first 10 chars): \(String(apiKey.prefix(10)))...")
        
        // Make the request
        let (data, response) = try await URLSession.shared.data(for: request)
        
        // Check response
        guard let httpResponse = response as? HTTPURLResponse else {
            throw OnlineSummarizationError.invalidResponse
        }
        
        print("🌐 Response status: \(httpResponse.statusCode)")
        
        guard httpResponse.statusCode == 200 else {
            // Try to extract error message
            let responseString = String(data: data, encoding: .utf8) ?? "No response body"
            print("❌ API Error Response: \(responseString)")
            
            if let errorJson = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let error = errorJson["error"] as? [String: Any],
               let message = error["message"] as? String {
                throw OnlineSummarizationError.apiError(message)
            }
            throw OnlineSummarizationError.httpError(httpResponse.statusCode)
        }
        
        // Parse response
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let choices = json["choices"] as? [[String: Any]],
              let firstChoice = choices.first,
              let message = firstChoice["message"] as? [String: Any],
              let content = message["content"] as? String else {
            throw OnlineSummarizationError.invalidResponse
        }
        
        // Stream the final result if callback is provided
        if let onProgress = onProgress {
            onProgress(content)
        }
        
        return content
    }
    
    /// Analyze new transcript and generate summary actions (agentic workflow)
    /// - Parameters:
    ///   - newText: New transcript text
    ///   - currentBulletPoints: Current bullet points in summary
    ///   - onProgress: Optional callback for streaming
    /// - Returns: Array of actions to perform
    public func analyzeAndDecide(newText: String, currentBulletPoints: [SummaryBulletPoint], onProgress: (@Sendable (String) -> Void)? = nil) async throws -> [SummaryAction] {
        // For now, use simple summarization and parse as append actions
        // In a more sophisticated version, we could prompt the API to analyze existing bullets
        let summary = try await summarize(newText, onProgress: onProgress)
        
        // Parse actions from the summary
        let actions = SummaryAction.parse(from: summary)
        
        print("🌐 OnlineSummarizationProvider: Parsed \(actions.count) actions from API response")
        
        return actions
    }
}

/// Errors that can occur with online summarization
public enum OnlineSummarizationError: LocalizedError {
    case invalidEndpoint
    case invalidResponse
    case apiError(String)
    case httpError(Int)
    case missingAPIKey
    
    public var errorDescription: String? {
        switch self {
        case .invalidEndpoint:
            return "Invalid API endpoint URL"
        case .invalidResponse:
            return "Invalid response from API"
        case .apiError(let message):
            return "API error: \(message)"
        case .httpError(let code):
            if code == 403 {
                return "HTTP 403 Forbidden: Check your API key has proper permissions and is valid. See console logs for details."
            } else if code == 401 {
                return "HTTP 401 Unauthorized: Invalid or missing API key"
            } else {
                return "HTTP error: \(code)"
            }
        case .missingAPIKey:
            return "API key not configured"
        }
    }
}
