//
//  SummaryService.swift
//  MeetingAssistantCore
//
//  Service for summarizing text into bullet points
//

import Foundation

/// Summarization provider type
public enum SummarizationProvider {
    case local  // On-device Core ML model
    case online // Online API (OpenAI, etc.)
}

/// Service that handles text summarization using a language model
class SummaryService: @unchecked Sendable {
    
    private let summarizationModel: SummarizationModel
    private var onlineProvider: OnlineSummarizationProvider?
    private var currentProvider: SummarizationProvider
    
    private var bulletPoints: [SummaryBulletPoint] = []
    private var nextID: Int = 1
    
    /// Check if summarization model is ready
    public var isReady: Bool {
        switch currentProvider {
        case .local:
            return summarizationModel.isModelLoaded
        case .online:
            return onlineProvider != nil
        }
    }
    
    public init(provider: SummarizationProvider = .local) {
        // Initialize model without auto-loading
        self.summarizationModel = SummarizationModel(autoLoad: false)
        self.currentProvider = provider
        
        // Initialize online provider if needed
        if provider == .online {
            self.configureOnlineProvider()
        } else {
            // Only load local model if starting with local provider
            self.summarizationModel.ensureModelLoaded()
        }
    }
    
    /// Switch to online API provider
    /// - Parameters:
    ///   - apiKey: API key for authentication
    ///   - endpoint: Optional custom endpoint (defaults to Nautilus)
    ///   - model: Optional model name (defaults to gemma3)
    public func switchToOnlineProvider(apiKey: String, endpoint: String? = nil, model: String? = nil) {
        let finalEndpoint = endpoint ?? "https://ellm.nrp-nautilus.io/v1/chat/completions"
        let finalModel = model ?? "gemma3"
        
        self.onlineProvider = OnlineSummarizationProvider(
            apiKey: apiKey,
            apiEndpoint: finalEndpoint,
            model: finalModel
        )
        self.currentProvider = .online
        print("✅ SummaryService: Switched to online provider (\(finalModel))")
    }
    
    /// Switch to local on-device model
    public func switchToLocalProvider() {
        self.currentProvider = .local
        // Ensure model is loaded when switching to local
        self.summarizationModel.ensureModelLoaded()
        print("✅ SummaryService: Switched to local on-device model")
    }
    
    /// Configure online provider from stored API key (if available)
    private func configureOnlineProvider() {
        if let apiKey = APIKeyManager.shared.getAPIKey() {
            let endpoint = APIKeyManager.shared.getAPIEndpoint() ?? "https://ellm.nrp-nautilus.io/v1/chat/completions"
            let model = APIKeyManager.shared.getModelName() ?? "gemma3"
            
            self.onlineProvider = OnlineSummarizationProvider(
                apiKey: apiKey,
                apiEndpoint: endpoint,
                model: model
            )
            print("✅ SummaryService: Configured online provider from stored credentials")
        }
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
        
        // Ask LLM to decide what actions to take (using configured provider)
        let actions: [SummaryAction]
        
        switch currentProvider {
        case .local:
            actions = try await summarizationModel.analyzeAndDecide(
                newText: newText,
                currentBulletPoints: bulletPoints,
                onProgress: onProgress
            )
        case .online:
            guard let provider = onlineProvider else {
                throw SummaryError.providerNotConfigured
            }
            actions = try await provider.analyzeAndDecide(
                newText: newText,
                currentBulletPoints: bulletPoints,
                onProgress: onProgress
            )
        }
        
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
        
        // Process through selected provider
        let summary: String
        
        switch currentProvider {
        case .local:
            summary = try await summarizationModel.summarize(text, onProgress: onProgress)
        case .online:
            guard let provider = onlineProvider else {
                throw SummaryError.providerNotConfigured
            }
            summary = try await provider.summarize(text, onProgress: onProgress)
        }
        
        return summary
    }
}

/// Errors that can occur during summarization
public enum SummaryError: LocalizedError {
    case emptyText
    case modelNotLoaded
    case providerNotConfigured
    case summarizationFailed(String)
    
    public var errorDescription: String? {
        switch self {
        case .emptyText:
            return "No text to summarize"
        case .modelNotLoaded:
            return "Summarization model is not loaded"
        case .providerNotConfigured:
            return "Online summarization provider is not configured"
        case .summarizationFailed(let reason):
            return "Summarization failed: \(reason)"
        }
    }
}
