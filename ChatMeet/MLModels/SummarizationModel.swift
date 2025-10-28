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
    
    public init() {
        // Initialize the summarization model
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
                
                // Use ModelLoader to load with GPU acceleration
                self.model = try LanguageModel.loadCompiled(url: modelURL, computeUnits: .cpuAndGPU)
                print("SummarizationModel: ✓ Successfully loaded Llama 3.2 model")

            } catch {
                print("SummarizationModel: ✗ Failed to load model: \(error)")
            }
        }
    }
    
    /// Find model URL in bundle
    private func findModelURL(named: String) -> URL? {
        // Try .mlpackage
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
        return nil
    }
    
    /// Generate bullet point summary from transcribed text
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
        
        print("SummarizationModel: Starting generation...")
        
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
            
            print("SummarizationModel: ✓ Generation complete")
            
            return summary
        } catch {
            print("SummarizationModel: ✗ Generation failed: \(error)")
            throw SummarizationError.generationFailed(error.localizedDescription)
        }
    }
    
    /// Create a Llama 3.2 Instruct formatted prompt
    /// - Parameter text: Input text to summarize
    /// - Returns: Formatted prompt with system message
    private func createLlamaPrompt(for text: String) -> String {
        // Llama 3.2 Instruct uses special tokens for chat format
        // <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        // {system_message}<|eot_id|>
        // <|start_header_id|>user<|end_header_id|>
        // {user_message}<|eot_id|>
        // <|start_header_id|>assistant<|end_header_id|>
        
        let systemMessage = """
        You are a helpful assistant that summarizes meeting transcripts into clear, concise bullet points. \
        Extract the key points, action items, and important decisions. \
        Format your response as bullet points using the • character.
        """
        
        let userMessage = """
        Please summarize the following meeting transcript into bullet points:
        
        \(text)
        """
        return "write a poem about cats"
        return """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        
        \(systemMessage)<|eot_id|><|start_header_id|>user<|end_header_id|>
        
        \(userMessage)<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        
        """
    }
    
    /// Extract the generated summary from model output
    /// - Parameters:
    ///   - output: Full model output
    ///   - prompt: Original prompt
    /// - Returns: Cleaned summary text
    private func extractSummary(from output: String, prompt: String) -> String {
        var summary = output
        
        // Debug: log what we're working with
        #if DEBUG
        print("SummarizationModel: Raw output length: \(output.count)")
        print("SummarizationModel: Prompt length: \(prompt.count)")
        print("SummarizationModel: Output starts with prompt: \(output.hasPrefix(prompt))")
        #endif
        
        // Remove the prompt prefix if present
        if summary.hasPrefix(prompt) {
            summary = String(summary.dropFirst(prompt.count))
            print("SummarizationModel: Removed exact prompt match")
        } else {
            // Try to find the assistant header as a fallback
            // Sometimes the model output doesn't include the full prompt
            let assistantHeader = "<|start_header_id|>assistant<|end_header_id|>"
            if let range = summary.range(of: assistantHeader) {
                summary = String(summary[range.upperBound...])
                print("SummarizationModel: Removed content up to assistant header")
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
        
        #if DEBUG
        print("SummarizationModel: Final summary length: \(summary.count)")
        if summary.count < 200 {
            print("SummarizationModel: Final summary: \(summary)")
        }
        #endif
        
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
