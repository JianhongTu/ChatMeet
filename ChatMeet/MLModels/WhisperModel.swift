//
//  WhisperModel.swift
//  MLModels
//
//  Wrapper for Whisper speech recognition model
//

import Foundation
@preconcurrency import CoreML
@preconcurrency import Tokenizers
import Accelerate

/// Context for streaming transcription to maintain state across chunks
class StreamingTranscriptionContext {    
    // Audio management
    var contextAudio: [Float] = []    // Audio from last window (overlap portion)
    
    // Statistics
    var totalTokensGenerated: Int = 0
    var chunkCount: Int = 0
    
    // Configuration - unified with Parakeet
    let windowDuration: Float = 20.0     // 20 seconds window
    let overlapDuration: Float = 2.0     // 2 seconds overlap
    let sampleRate: Float = 16000.0
    
    func reset() {
        contextAudio = []
        totalTokensGenerated = 0
        chunkCount = 0
    }
}

/// Manages the Whisper Core ML model for speech-to-text transcription
class WhisperModel: @unchecked Sendable {
    
    private var encoderModel: MLModel?
    private var decoderModel: MLModel?
    private var tokenizer: Tokenizer?
    
    // Model configuration
    private let modelName = "openai/whisper-tiny"
    private let encoderModelName = "WhisperEncoder_whisper-tiny"
    private let decoderModelName = "WhisperDecoder_whisper-tiny"
    
    // Audio configuration
    private let sampleRate: Float = 16000.0
    private let audioLength = 30  // 30 seconds
    private let expectedSamples = 480000  // 16kHz * 30s
    
    // Model readiness state
    private(set) var isReady: Bool = false
    
    // Compute backend configuration
    private var currentBackend: ComputeBackend = .all
    
    public init() {
        // Models will be loaded lazily via loadModel()
    }
    
    /// Get the current compute backend
    public var computeBackend: ComputeBackend {
        return currentBackend
    }
    
    /// Set the compute backend for model inference
    /// - Parameter backend: The desired compute backend
    /// - Note: Models will be reloaded automatically if already loaded
    public func setComputeBackend(_ backend: ComputeBackend) async throws {
        print("WhisperModel: 🔄 Switching compute backend to \(backend.description)")
        
        // Store the new backend
        currentBackend = backend
        
        // If models are already loaded, reload them with new configuration
        if isReady {
            unloadModel()
            try await loadModel()
        }
    }
    
    /// Load the Core ML Whisper encoder and decoder models
    /// This should be called after user selects the Whisper model
    public func loadModel() async throws {
        print("WhisperModel: 🔄 Starting model loading with \(currentBackend.description) backend...")
        
        try await loadModels()
        try await loadTokenizer()
        
        isReady = true
        print("WhisperModel: ✅ Models loaded successfully")
    }
    
    /// Unload all models to free memory
    public func unloadModel() {
        encoderModel = nil
        decoderModel = nil
        tokenizer = nil
        isReady = false
        print("WhisperModel: 🗑️ Models unloaded")
    }
    
    /// Load the Core ML Whisper encoder and decoder models
    private func loadModels() async throws {
        // Configure Core ML with selected compute backend
        let config = MLModelConfiguration()
        config.computeUnits = currentBackend.mlComputeUnits
        
        // Load encoder model on background thread to prevent UI freeze
        guard let encoderURL = findModelURL(named: encoderModelName) else {
            throw WhisperError.modelNotFound(encoderModelName)
        }
        encoderModel = try await Task.detached(priority: .userInitiated) {
            try MLModel(contentsOf: encoderURL, configuration: config)
        }.value
        print("WhisperModel: ✓ Loaded encoder model")
        
        // Load decoder model on background thread to prevent UI freeze
        guard let decoderURL = findModelURL(named: decoderModelName) else {
            throw WhisperError.modelNotFound(decoderModelName)
        }
        decoderModel = try await Task.detached(priority: .userInitiated) {
            try MLModel(contentsOf: decoderURL, configuration: config)
        }.value
        print("WhisperModel: ✓ Loaded decoder model")
    }
    
    /// Load tokenizer from Hugging Face
    private func loadTokenizer() async throws {
        do {
            tokenizer = try await AutoTokenizer.from(pretrained: modelName)
            print("WhisperModel: ✓ Loaded tokenizer from \(modelName)")
        } catch {
            print("WhisperModel: ✗ Failed to load tokenizer: \(error.localizedDescription)")
            throw WhisperError.tokenizerLoadFailed(error.localizedDescription)
        }
    }
    
    /// Find model URL in bundle or file system
    /// - Parameter named: Model name without extension
    /// - Returns: URL to compiled model if found
    private func findModelURL(named: String) -> URL? {
        // Try to find in main bundle resources (compiled .mlmodelc for release builds)
        if let url = Bundle.main.url(forResource: named, withExtension: "mlmodelc") {
            return url
        }
        
        // Try to find .mlpackage in bundle
        if let url = Bundle.main.url(forResource: named, withExtension: "mlpackage") {
            return url
        }
        
        // Try without extension (sometimes Xcode compiles differently)
        if let url = Bundle.main.url(forResource: named, withExtension: nil) {
            return url
        }
        
        // For development: Try Models directory at project root
        // This is relative to the bundle's parent directory structure
        let bundleURL = Bundle.main.bundleURL.deletingLastPathComponent().deletingLastPathComponent()
        let modelsPath = bundleURL.appendingPathComponent("Models/\(named).mlpackage")
        if FileManager.default.fileExists(atPath: modelsPath.path) {
            print("WhisperModel: ✓ Found model in Models/ directory: \(modelsPath.path)")
            return modelsPath
        }
        
        return nil
    }
    
    /// Transcribe audio data to text
    /// - Parameters:
    ///   - audioData: Raw audio data in WAV format (16kHz, mono, PCM)
    ///   - onProgress: Optional callback for real-time token streaming
    /// - Returns: Final transcribed text
    public func transcribe(_ audioData: Data, onProgress: (@Sendable (String) -> Void)? = nil) async throws -> String {
        let transcribeStartTime = CFAbsoluteTimeGetCurrent()
        
        // Wait for models to be loaded
        guard let encoder = encoderModel else {
            throw WhisperError.modelNotLoaded
        }
        
        // Extract PCM samples from WAV data using AudioPreprocessor
        let preprocessStartTime = CFAbsoluteTimeGetCurrent()
        let samples = try AudioPreprocessor.extractPCMSamples(from: audioData)
        
        // Pad/trim to expected length
        let paddedSamples = AudioPreprocessor.padOrTrim(samples, to: expectedSamples)
        let audioInput = try convertToMLMultiArray(paddedSamples)
        let preprocessTime = CFAbsoluteTimeGetCurrent() - preprocessStartTime
        print("⏱️ WhisperModel Preprocessing: \(String(format: "%.3f", preprocessTime))s")
        
        // Run encoder to get audio features
        let encoderOutput = try runEncoder(encoder, input: audioInput)
        
        // Run decoder with tokenizer to generate text
        guard let decoder = decoderModel, let tokenizer = tokenizer else {
            throw WhisperError.modelNotLoaded
        }
        
        let transcription = try await runDecoder(decoder, encoderOutput: encoderOutput, tokenizer: tokenizer, onProgress: onProgress)
        
        let totalTranscribeTime = CFAbsoluteTimeGetCurrent() - transcribeStartTime
        print("⏱️ WhisperModel Total Transcription: \(String(format: "%.3f", totalTranscribeTime))s")
        
        return transcription
    }
    
    /// Transcribe audio incrementally with audio overlap strategy (unified with Parakeet)
    /// - Parameters:
    ///   - audioData: Audio data (30s ring buffer window)
    ///   - context: Streaming context with audio overlap
    ///   - onProgress: Optional callback for streaming token updates
    /// - Returns: Only the NEW transcription text from this chunk
    /// - Note: Uses audio overlap strategy matching Parakeet:
    ///         - Window: 20 seconds of audio
    ///         - Overlap: 2 seconds with previous window
    ///         - Deduplication: Word-based overlap removal
    public func transcribeIncremental(audioData: Data, context: StreamingTranscriptionContext, onProgress: (@Sendable (String) -> Void)? = nil) async throws -> String {
        guard let encoder = encoderModel, let decoder = decoderModel, let tokenizer = tokenizer else {
            throw WhisperError.modelNotLoaded
        }
        
        context.chunkCount += 1
        #if DEBUG
        print("🔵 WhisperModel: Chunk #\(context.chunkCount) - 20s window with 2s overlap")
        #endif
        
        // Extract PCM samples from WAV data (full 30s ring buffer) using AudioPreprocessor
        let samples = try AudioPreprocessor.extractPCMSamples(from: audioData)
        
        // Extract window: overlap (2s) + new audio (18s) = 20s total
        let windowSamples = Int(context.windowDuration * context.sampleRate)      // 20s = 320,000 samples
        let overlapSamples = Int(context.overlapDuration * context.sampleRate)    // 2s = 32,000 samples
        let newAudioSamples = windowSamples - overlapSamples                      // 18s = 288,000 samples
        
        // Get the newest audio from the ring buffer
        let startIndex = max(0, samples.count - newAudioSamples)
        let newAudio = Array(samples[startIndex..<samples.count])
        
        // Combine overlap from previous window + new audio
        let overlapToUse = context.contextAudio.suffix(overlapSamples)
        let windowAudio = Array(overlapToUse) + newAudio
        
        #if DEBUG
        print("🔵 WhisperModel: Overlap audio: \(overlapToUse.count) samples (~\(Float(overlapToUse.count)/context.sampleRate)s)")
        print("🔵 WhisperModel: New audio: \(newAudio.count) samples (~\(Float(newAudio.count)/context.sampleRate)s)")
        print("🔵 WhisperModel: Window total: \(windowAudio.count) samples (~\(Float(windowAudio.count)/context.sampleRate)s)")
        #endif
        
        // Pad to 30 seconds (480,000 samples) - put window at END with leading zeros
        let paddedSamples = AudioPreprocessor.padOrTrim(windowAudio, to: expectedSamples)
        
        // Convert to MLMultiArray and run encoder
        let audioInput = try convertToMLMultiArray(paddedSamples)
        let encoderOutput = try runEncoder(encoder, input: audioInput)
        
        // Run decoder WITHOUT context tokens (fresh prompt each time)
        let transcription = try await runDecoder(decoder, encoderOutput: encoderOutput, tokenizer: tokenizer, onProgress: onProgress)
        
        // Update context audio for next chunk (save current new audio as overlap)
        context.contextAudio = newAudio
        
        // Update statistics
        context.totalTokensGenerated += transcription.split(separator: " ").count
        
        #if DEBUG
        print("🔵 WhisperModel: Chunk #\(context.chunkCount) complete - transcription: '\(transcription)'")
        #endif
        
        return transcription
    }
    
    // MARK: - Audio Preprocessing
    // Note: Basic audio preprocessing (WAV parsing, PCM decoding, resampling)
    // has been moved to AudioPreprocessor.swift for reuse across models
    
    /// Convert float array to MLMultiArray [1, N] using MLShapedArray
    private func convertToMLMultiArray(_ samples: [Float]) throws -> MLMultiArray {
        // Use MLShapedArray for efficient conversion
        let shaped = MLShapedArray(scalars: samples, shape: [1, samples.count])
        return MLMultiArray(shaped)
    }
    
    // MARK: - Model Inference
    
    /// Run encoder model on raw audio waveform
    /// - Parameters:
    ///   - encoder: Encoder MLModel
    ///   - input: Raw audio waveform as MLMultiArray [1, 480000]
    /// - Returns: Encoder output features (encoderHiddenStates)
    private func runEncoder(_ encoder: MLModel, input: MLMultiArray) throws -> MLMultiArray {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Create input feature provider
        let inputName = "audioInput"
        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(multiArray: input)])
        
        // Run inference
        let output = try encoder.prediction(from: inputFeatures)
        
        // Extract encoder output
        let outputName = "encoderHiddenStates"
        guard let encoderOutput = output.featureValue(for: outputName)?.multiArrayValue else {
            throw WhisperError.inferenceFailed("Failed to extract encoder output")
        }
        
        let encoderTime = CFAbsoluteTimeGetCurrent() - startTime
        print("⏱️ WhisperModel Encoder: \(String(format: "%.3f", encoderTime))s")
        
        return encoderOutput
    }
    
    /// Run decoder model to generate text tokens
    /// - Parameters:
    ///   - decoder: Decoder MLModel
    ///   - encoderOutput: Output from encoder
    ///   - tokenizer: Tokenizer for decoding
    /// - Returns: Transcribed text
    private func runDecoder(_ decoder: MLModel, encoderOutput: MLMultiArray, tokenizer: Tokenizer, onProgress: (@Sendable (String) -> Void)? = nil) async throws -> String {
        let totalStartTime = CFAbsoluteTimeGetCurrent()
        
        // Special tokens for Whisper
        // Whisper uses specific token IDs - these may need adjustment based on actual model
        let startTokenId = 50258  // <|startoftranscript|>
        let eotTokenId = 50257    // <|endoftext|>

        // Standard prompt: <|startoftranscript|> <|en|> <|transcribe|> <|no_timestamps|>
        // Replace 50259 if you want dynamic language tokens
        let promptTokens: [Int] = [startTokenId, 50259, 50359, 50363]
        var generatedTokens: [Int] = promptTokens
        let maxTokens = 448  // Whisper's max sequence length
        
        // Create state for stateful model (KV cache management)
        let state = decoder.makeState()
        
        var isPrefill = true
        
        // Repetition detection: track consecutive identical tokens
        var repetitionCount = 0
        let maxRepetitions = 5  // Stop if same token repeats >5 times
        
        // Timing metrics
        var totalPrepareInputTime: Double = 0
        var totalInferenceTime: Double = 0
        var totalTokenizerTime: Double = 0
        var iterationCount = 0

        // Autoregressive decoding
        for _ in 0..<maxTokens {
            // Check if we're about to exceed max tokens BEFORE preparing input
            if generatedTokens.count >= maxTokens {
                #if DEBUG
                print("⚠️ WhisperModel: Reached max token limit (\(maxTokens))")
                #endif
                break
            }
            
            // Prepare decoder input
            let prepareStartTime = CFAbsoluteTimeGetCurrent()
            let decoderInput = try prepareDecoderInput(
                tokens: generatedTokens,
                encoderOutput: encoderOutput,
                isPrefill: isPrefill
            )
            totalPrepareInputTime += CFAbsoluteTimeGetCurrent() - prepareStartTime
            
            // Run decoder inference with state
            // The state maintains the KV cache across iterations
            let inferenceStartTime = CFAbsoluteTimeGetCurrent()
            let output = try await decoder.prediction(from: decoderInput, using: state)
            totalInferenceTime += CFAbsoluteTimeGetCurrent() - inferenceStartTime
            
            iterationCount += 1
            
            // Get next token ID directly from model output
            guard let nextToken = extractMaxTokenId(from: output) else {
                throw WhisperError.inferenceFailed("Failed to extract max token ID")
            }
            
            // Check for end of transcription
            if nextToken == eotTokenId {
                break
            }
            
            // Repetition detection: check if this token is same as previous
            if generatedTokens.count > promptTokens.count {
                let prevToken = generatedTokens[generatedTokens.count - 1]
                if nextToken == prevToken {
                    repetitionCount += 1
                    if repetitionCount >= maxRepetitions {
                        #if DEBUG
                        print("⚠️ WhisperModel: Detected repetition loop (token \(nextToken) repeated \(repetitionCount + 1) times) - stopping generation")
                        #endif
                        break
                    }
                } else {
                    repetitionCount = 0  // Reset counter when token changes
                }
            }
            
            generatedTokens.append(nextToken)
            isPrefill = false
            
            // Stream partial transcription if callback is provided
            // Throttle: only decode every 10 tokens to avoid expensive tokenizer calls
            if let onProgress = onProgress, generatedTokens.count % 10 == 0 {
                let tokenizerStartTime = CFAbsoluteTimeGetCurrent()
                let partialTranscription = decodeTokens(generatedTokens, using: tokenizer)
                let cleanedPartial = deduplicateRepetitions(partialTranscription)
                totalTokenizerTime += CFAbsoluteTimeGetCurrent() - tokenizerStartTime
                
                // Dispatch to main thread asynchronously to avoid blocking decoding
                Task { @MainActor in
                    onProgress(cleanedPartial)
                }
            }
        }
        
        // Decode tokens to text (final)
        let tokenizerStartTime = CFAbsoluteTimeGetCurrent()
        let transcription = decodeTokens(generatedTokens, using: tokenizer)
        totalTokenizerTime += CFAbsoluteTimeGetCurrent() - tokenizerStartTime
        
        // Remove any repeated phrases from the output
        let cleanedTranscription = deduplicateRepetitions(transcription)
        
        let totalTime = CFAbsoluteTimeGetCurrent() - totalStartTime
        
        // Print timing breakdown
        print("⏱️ WhisperModel Decoder Timing:")
        print("   Total iterations: \(iterationCount)")
        print("   Prepare input: \(String(format: "%.3f", totalPrepareInputTime))s (\(String(format: "%.1f", totalPrepareInputTime/totalTime*100))%)")
        print("   Inference: \(String(format: "%.3f", totalInferenceTime))s (\(String(format: "%.1f", totalInferenceTime/totalTime*100))%)")
        print("   Tokenizer: \(String(format: "%.3f", totalTokenizerTime))s (\(String(format: "%.1f", totalTokenizerTime/totalTime*100))%)")
        print("   Total: \(String(format: "%.3f", totalTime))s")
        print("   Avg per iteration: \(String(format: "%.3f", totalTime/Double(iterationCount)))s")
        
        #if DEBUG
        if transcription != cleanedTranscription {
            print("⚠️ WhisperModel: Removed repetitions: \(transcription.count - cleanedTranscription.count) chars")
        }
        #endif
        
        // Send final update if we didn't just send one
        if let onProgress = onProgress, generatedTokens.count % 10 != 0 {
            Task { @MainActor in
                onProgress(cleanedTranscription)
            }
        }
        
        return cleanedTranscription
    }
    
    /// Prepare decoder input from tokens and encoder output
    /// - Parameters:
    ///   - tokens: Generated token IDs (all tokens generated so far)
    ///   - encoderOutput: Encoder output features
    /// - Returns: Feature provider for decoder
    private func prepareDecoderInput(tokens: [Int], encoderOutput: MLMultiArray, isPrefill: Bool) throws -> MLFeatureProvider {
        if isPrefill {
            // Feed the entire prompt sequence to build KV cache
            let seqLen = tokens.count
            
            // Convert tokens using MLShapedArray
            let tokenArray = MLMultiArray(MLShapedArray(scalars: tokens.map { Int32($0) }, shape: [1, seqLen]))

            // Causal mask: lower-triangular for self-attention during prefill
            // Create mask efficiently: all zeros with -inf in upper triangle
            let maskSize = seqLen * seqLen
            var maskValues = [Float](repeating: 0.0, count: maskSize)
            
            // Set upper triangle to -inf using stride for better cache locality
            for i in 0..<seqLen {
                let rowStart = i * seqLen + i + 1
                if rowStart < (i + 1) * seqLen {
                    let upperTriangleCount = seqLen - i - 1
                    vDSP_vfill([-Float.infinity], &maskValues[rowStart], 1, vDSP_Length(upperTriangleCount))
                }
            }
            
            let causalMask = MLMultiArray(MLShapedArray(scalars: maskValues, shape: [1, 1, seqLen, seqLen]))

            let inputDict: [String: MLFeatureValue] = [
                "inputIds": MLFeatureValue(multiArray: tokenArray),
                "encoderHiddenStates": MLFeatureValue(multiArray: encoderOutput),
                "causalMask": MLFeatureValue(multiArray: causalMask)
            ]
            return try MLDictionaryFeatureProvider(dictionary: inputDict)
        } else {
            // Extension step: feed only the last token
            // Reuse single-element arrays - create once per call is still fast enough
            let lastToken = Int32(tokens.last!)
            let tokenArray = MLMultiArray(MLShapedArray(scalars: [lastToken], shape: [1, 1]))

            // With KV cache, mask is all zeros - grow as needed
            // This is lightweight since it's just zeros
            let pastSeqLen = tokens.count
            let causalMask = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: pastSeqLen)], dataType: .float32)
            // All zeros by default, no need to set

            let inputDict: [String: MLFeatureValue] = [
                "inputIds": MLFeatureValue(multiArray: tokenArray),
                "encoderHiddenStates": MLFeatureValue(multiArray: encoderOutput),
                "causalMask": MLFeatureValue(multiArray: causalMask)
            ]
            return try MLDictionaryFeatureProvider(dictionary: inputDict)
        }
    }
    
    /// Extract max token ID from decoder output
    /// - Parameter output: Decoder output features
    /// - Returns: Token ID with maximum probability
    private func extractMaxTokenId(from output: MLFeatureProvider) -> Int? {
        let outputName = "tokenId"
        guard let tokenIdValue = output.featureValue(for: outputName)?.multiArrayValue else {
            return nil
        }
        
        // Extract the token ID (should be a scalar or [1] array)
        return tokenIdValue[0].intValue
    }
    
    /// Decode token IDs to text using tokenizer
    /// - Parameters:
    ///   - tokens: Array of token IDs
    ///   - tokenizer: Tokenizer instance
    ///   - skipPrompt: Whether to skip the first 4 prompt tokens (default true)
    /// - Returns: Decoded text string
    private func decodeTokens(_ tokens: [Int], using tokenizer: Tokenizer, skipPrompt: Bool = true) -> String {
        // Remove prompt tokens if needed: <|startoftranscript|> <|en|> <|transcribe|> <|no_timestamps|>
        // The first 4 tokens are the prompt, so skip them
        let textTokens: [Int]
        if skipPrompt {
            let promptLength = 4
            textTokens = tokens.count > promptLength ? Array(tokens.dropFirst(promptLength)) : []
        } else {
            textTokens = tokens
        }
        
        // Decode using tokenizer
        let decoded = tokenizer.decode(tokens: textTokens)
        
        // Remove any special tokens that might still appear in the decoded text
        var cleaned = decoded
        let specialTokens = [
            "<|startoftranscript|>",
            "<|en|>",
            "<|transcribe|>",
            "<|notimestamps|>",
            "<|nocaptions|>",
            "<|endoftext|>"
        ]
        
        for token in specialTokens {
            cleaned = cleaned.replacingOccurrences(of: token, with: "")
        }
        
        return cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    
    /// Remove repeated phrases and words from transcription
    /// - Parameter text: Transcribed text potentially with repetitions
    /// - Returns: Text with repetitions removed
    private func deduplicateRepetitions(_ text: String) -> String {
        guard !text.isEmpty else { return text }
        
        // Split into words while preserving punctuation
        let words = text.split(separator: " ").map(String.init)
        guard words.count > 1 else { return text }
        
        var result: [String] = []
        var i = 0
        
        while i < words.count {
            let currentWord = words[i]
            
            // Look ahead to find if there's a repetitive pattern
            var patternLength = 1
            let maxPatternLength = min(20, (words.count - i) / 2)  // Don't look for patterns longer than 20 words
            var foundRepetition = false
            
            // Try to find repeating patterns starting from longer sequences
            for length in stride(from: maxPatternLength, through: 1, by: -1) {
                if i + length * 2 <= words.count {
                    // Check if the pattern repeats
                    let pattern = words[i..<(i + length)]
                    let nextSegment = words[(i + length)..<min(i + length * 2, words.count)]
                    
                    // Compare normalized versions (case-insensitive, trim punctuation)
                    let patternNormalized = pattern.map { $0.lowercased().trimmingCharacters(in: .punctuationCharacters) }
                    let nextNormalized = nextSegment.map { $0.lowercased().trimmingCharacters(in: .punctuationCharacters) }
                    
                    if patternNormalized == Array(nextNormalized.prefix(length)) {
                        patternLength = length
                        foundRepetition = true
                        
                        #if DEBUG
                        print("🔍 WhisperModel: Found repetition of \(length) words: '\(pattern.joined(separator: " "))'")
                        #endif
                        break
                    }
                }
            }
            
            if foundRepetition {
                // Add the pattern once, skip all repetitions
                result.append(contentsOf: words[i..<(i + patternLength)])
                
                // Count how many times it repeats and skip them all
                var skipCount = patternLength
                while i + skipCount + patternLength <= words.count {
                    let nextPattern = words[(i + skipCount)..<(i + skipCount + patternLength)]
                    let pattern = words[i..<(i + patternLength)]
                    
                    let nextNormalized = nextPattern.map { $0.lowercased().trimmingCharacters(in: .punctuationCharacters) }
                    let patternNormalized = pattern.map { $0.lowercased().trimmingCharacters(in: .punctuationCharacters) }
                    
                    if nextNormalized == patternNormalized {
                        skipCount += patternLength
                    } else {
                        break
                    }
                }
                
                i += skipCount
            } else {
                // No repetition found, add current word and move on
                result.append(currentWord)
                i += 1
            }
        }
        
        return result.joined(separator: " ")
    }
}

/// Errors that can occur during Whisper operations
public enum WhisperError: LocalizedError {
    case modelNotLoaded
    case modelNotFound(String)
    case tokenizerLoadFailed(String)
    case preprocessingFailed
    case inferenceFailed(String)
    
    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "Whisper model is not loaded"
        case .modelNotFound(let name):
            return "Model not found: \(name)"
        case .tokenizerLoadFailed(let reason):
            return "Failed to load tokenizer: \(reason)"
        case .preprocessingFailed:
            return "Failed to preprocess audio"
        case .inferenceFailed(let reason):
            return "Inference failed: \(reason)"
        }
    }
}
