//
//  SummarizationModel.swift
//  MLModels
//
//  Wrapper for small language model for text summarization
//

import Foundation
import CoreML
import Tokenizers
import Generation
import Models

/// Manages the language model for text summarization into bullet points
class SummarizationModel: @unchecked Sendable {
    
    private var model: LanguageModel?
    
    // Model configuration
    private let modelName = "meta-llama/Llama-3.2-1B-Instruct"
    private let compiledModelName = "StatefulLlama3.2Instruct"
    
    /// Check if model is loaded and ready
    public var isModelLoaded: Bool {
        return model != nil
    }
    
    public init(autoLoad: Bool = true) {
        // Only auto-load if requested
        if autoLoad {
            loadModel()
        }
    }
    
    /// Manually trigger model loading
    public func ensureModelLoaded() {
        guard model == nil else { return }
        loadModel()
    }
    
    /// Load the Core ML language model and tokenizer
    private func loadModel() {
        // Load model using ModelLoader with fixed URL
        Task { [weak self] in
            guard let self = self else { return }
            
            do {
                // Find the model URL in bundle
                guard let modelURL = self.findModelURL(named: self.compiledModelName) else {
                    print("SummarizationModel: ✗ Model not found")
                    print("SummarizationModel: Looking for: \(self.compiledModelName).mlpackage")
                    return
                }
                
                // Use ModelLoader to load with all available accelerators (CPU, GPU, Neural Engine)
                self.model = try LanguageModel.loadCompiled(url: modelURL, computeUnits: .all)
                print("SummarizationModel: ✓ Successfully loaded Llama 3.2 model")

            } catch {
                print("SummarizationModel: ✗ Failed to load model: \(error)")
            }
        }
    }
    
    /// Find model URL in bundle
    private func findModelURL(named: String) -> URL? {
        // Try .mlpackage in bundle
        if let url = Bundle.main.url(forResource: named, withExtension: "mlpackage") {
            return url
        }
        // Try .mlmodelc (compiled)
        if let url = Bundle.main.url(forResource: named, withExtension: "mlmodelc") {
            return url
        }
        // Try without extension
        if let url = Bundle.main.url(forResource: named, withExtension: nil) {
            return url
        }
        
        // For development: Try Models directory at project root
        // This is relative to the bundle's parent directory structure
        let bundleURL = Bundle.main.bundleURL.deletingLastPathComponent().deletingLastPathComponent()
        let modelsPath = bundleURL.appendingPathComponent("Models/\(named).mlpackage")
        if FileManager.default.fileExists(atPath: modelsPath.path) {
            print("SummarizationModel: ✓ Found model in Models/ directory: \(modelsPath.path)")
            return modelsPath
        }
        
        return nil
    }
    
    /// Agentic workflow: Analyze new transcript and decide summary actions
    /// - Parameters:
    ///   - newText: New transcript text to analyze
    ///   - currentBulletPoints: Current bullet points in summary
    ///   - onProgress: Optional callback for real-time streaming
    /// - Returns: Array of actions to perform on the summary
    public func analyzeAndDecide(newText: String, currentBulletPoints: [SummaryBulletPoint], onProgress: (@Sendable (String) -> Void)? = nil) async throws -> [SummaryAction] {
        guard let model = model else {
            throw SummarizationError.modelNotLoaded
        }
        
        // Create agentic prompt
        let prompt = createAgenticPrompt(newText: newText, currentBulletPoints: currentBulletPoints)
        
        // DEBUG: Print full prompt
        print(String(repeating: "=", count: 80))
        print("🤖 SummarizationModel: INPUT PROMPT TO LLM:")
        print(String(repeating: "-", count: 80))
        print(prompt)
        print(String(repeating: "=", count: 80))
        
        // Configure generation - allow more tokens for actions without sampling
        let generationConfig = GenerationConfig(maxNewTokens: 256)
        
        do {
            // Generate actions with streaming
            let output = try await model.generate(
                config: generationConfig,
                prompt: prompt
            ) { partialResult in
                if let onProgress = onProgress {
                    let cleaned = self.extractSummary(from: partialResult, prompt: prompt)
                    onProgress(cleaned)
                }
            }
            
            // Extract and parse actions
            let actionText = extractSummary(from: output, prompt: prompt)
            
            // DEBUG: Print full LLM output
            print(String(repeating: "=", count: 80))
            print("🤖 SummarizationModel: RAW LLM OUTPUT:")
            print(String(repeating: "-", count: 80))
            print(output)
            print(String(repeating: "-", count: 80))
            print("🤖 SummarizationModel: CLEANED OUTPUT:")
            print(String(repeating: "-", count: 80))
            print(actionText)
            print(String(repeating: "=", count: 80))
            
            let actions = SummaryAction.parse(from: actionText)
            print("🤖 SummarizationModel: Parsed \(actions.count) actions: \(actions)")
            
            return actions
        } catch {
            print("SummarizationModel: ✗ Generation failed: \(error)")
            throw SummarizationError.generationFailed(error.localizedDescription)
        }
    }
    
    /// Generate bullet point summary from transcribed text (legacy method)
    /// - Parameters:
    ///   - text: Transcribed text to summarize
    ///   - onProgress: Optional callback for real-time token streaming
    /// - Returns: Bullet point summary
    public func summarize(_ text: String, onProgress: (@Sendable (String) -> Void)? = nil) async throws -> String {
        guard let model = model else {
            throw SummarizationError.modelNotLoaded
        }
        
        // Create Llama 3.2 Instruct prompt with system message
        let prompt = createLlamaPrompt(for: text)
        
        // Configure generation parameters - constrained to ~128 words
        let generationConfig = GenerationConfig(maxNewTokens: 128)
        
        do {
            // Generate summary with real-time callback
            let output = try await model.generate(
                config: generationConfig,
                prompt: prompt
            ) { partialResult in
                // Stream each new token to the callback
                if let onProgress = onProgress {
                    let cleaned = self.extractSummary(from: partialResult, prompt: prompt)
                    onProgress(cleaned)
                }
            }
            
            // Extract final generated text (removing prompt)
            let summary = extractSummary(from: output, prompt: prompt)
            
            return summary
        } catch {
            print("SummarizationModel: ✗ Generation failed: \(error)")
            throw SummarizationError.generationFailed(error.localizedDescription)
        }
    }
    
    /// Create a Llama 3.2 Instruct formatted prompt (legacy for batch summarization)
    /// - Parameter text: Input text to summarize
    /// - Returns: Formatted prompt with system message
    private func createLlamaPrompt(for text: String) -> String {
        return createAgenticPrompt(newText: text, currentBulletPoints: [])
    }
    
    /// Create a Llama 3.2 Instruct formatted prompt for agentic summary management
    /// - Parameters:
    ///   - newText: New transcript text to analyze
    ///   - currentBulletPoints: Existing bullet points in the summary
    /// - Returns: Formatted prompt with system message
    private func createAgenticPrompt(newText: String, currentBulletPoints: [SummaryBulletPoint]) -> String {
        let systemMessage = """
        You are a helpful meeting assistant. Summarize the key points from meeting transcripts. \
        Format your response as bullet points using markdown style.
        """
        
        let userMessage = """
        Create a high-level summary of the provided transcript with one item on each line.
        
        \(newText)
        """
        
        return """
        <|start_header_id|>system<|end_header_id|>
        \(systemMessage)<|eot_id|><|start_header_id|>user<|end_header_id|>
        
        \(userMessage)<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        
        • 
        """
        // Note: do not include BOS token here. It is automatically added by the tokenizer.
    }
    
    /// Extract the generated summary from model output
    /// - Parameters:
    ///   - output: Full model output
    ///   - prompt: Original prompt
    /// - Returns: Cleaned summary text
    private func extractSummary(from output: String, prompt: String) -> String {
        var summary = output
        
        // Remove the prompt prefix if present
        if summary.hasPrefix(prompt) {
            summary = String(summary.dropFirst(prompt.count))
        } else {
            // Try to find the assistant header as a fallback
            // Sometimes the model output doesn't include the full prompt
            let assistantHeader = "<|start_header_id|>assistant<|end_header_id|>"
            if let range = summary.range(of: assistantHeader) {
                summary = String(summary[range.upperBound...])
            }
        }
        
        // Remove all special tokens
        summary = summary.replacingOccurrences(of: "<|eot_id|>", with: "")
        summary = summary.replacingOccurrences(of: "<|end_of_text|>", with: "")
        summary = summary.replacingOccurrences(of: "<|begin_of_text|>", with: "")
        summary = summary.replacingOccurrences(of: "<|start_header_id|>system<|end_header_id|>", with: "")
        summary = summary.replacingOccurrences(of: "<|start_header_id|>user<|end_header_id|>", with: "")
        summary = summary.replacingOccurrences(of: "<|start_header_id|>assistant<|end_header_id|>", with: "")
        
        // Trim whitespace and newlines
        summary = summary.trimmingCharacters(in: .whitespacesAndNewlines)
        
        return summary
    }
}

/// Errors that can occur during summarization
public enum SummarizationError: LocalizedError {
    case modelNotLoaded
    case generationFailed(String)
    
    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "Summarization model or tokenizer is not loaded"
        case .generationFailed(let reason):
            return "Generation failed: \(reason)"
        }
    }
}
