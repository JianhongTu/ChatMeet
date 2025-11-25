//
//  ParakeetModel.swift
//  MLModels
//
//  Wrapper for Parakeet TDT v3 speech recognition model
//

import Foundation
@preconcurrency import CoreML
@preconcurrency import Tokenizers

/// Available compute backends for Core ML models
public enum ComputeBackend: String, CaseIterable, Sendable {
    case cpu = "CPU"
    case gpu = "GPU"
    case neuralEngine = "Neural Engine"
    case all = "Automatic"
    
    var mlComputeUnits: MLComputeUnits {
        switch self {
        case .cpu:
            return .cpuOnly
        case .gpu:
            return .cpuAndGPU
        case .neuralEngine:
            return .cpuAndNeuralEngine
        case .all:
            return .all
        }
    }
    
    var description: String {
        return self.rawValue
    }
}

/// Manages the Parakeet Core ML model for speech-to-text transcription
/// Parakeet uses a 3-component RNN-T architecture:
/// 1. Preprocessor: Audio -> Mel spectrogram features
/// 2. Encoder: Mel features -> Acoustic embeddings
/// 3. DecoderJoiner: Combined decoder+joiner -> Token predictions with argmax
class ParakeetModel: @unchecked Sendable {
    
    // Three Core ML models for optimized RNN-T architecture
    private var preprocessorModel: MLModel?
    private var encoderModel: MLModel?
    private var decoderJoinerModel: MLModel?
    
    private var tokenizer: Tokenizer?
    
    // Model configuration
    private let modelName = "ToviTu/parakeet-tdt-0.6b-v3-coreml"
    private let preprocessorModelName = "ParakeetPreprocessor"
    private let encoderModelName = "ParakeetEncoder"
    private let decoderJoinerModelName = "ParakeetDecoderJointer"
    
    // Audio configuration
    private let sampleRate: Float = 16000.0
    
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
    
    /// Decode token IDs to text using the loaded tokenizer
    /// - Parameter tokens: Array of token IDs to decode
    /// - Returns: Decoded text string
    public func decodeTokens(_ tokens: [Int]) -> String {
        guard let tokenizer = tokenizer else {
            return ""
        }
        let text = tokenizer.decode(tokens: tokens)
        return cleanParakeetArtifacts(text)
    }
    
    /// Set the compute backend for model inference
    /// - Parameter backend: The desired compute backend
    /// - Note: Models will be reloaded automatically if already loaded
    public func setComputeBackend(_ backend: ComputeBackend) async throws {
        // Store the new backend
        currentBackend = backend
        
        // If models are already loaded, reload them with new configuration
        if isReady {
            unloadModel()
            try await loadModel()
        }
    }
    
    /// Load all Parakeet Core ML models and tokenizer
    /// This should be called after user selects the Parakeet model
    public func loadModel() async throws {
        // Configure Core ML with selected compute backend
        let config = MLModelConfiguration()
        config.computeUnits = currentBackend.mlComputeUnits
        
        // Load all three models
        try await loadPreprocessor(config: config)
        try await loadEncoder(config: config)
        try await loadDecoderJoiner(config: config)
        
        // Load tokenizer from Hugging Face
        try await loadTokenizer()
        
        isReady = true
    }
    
    /// Unload all models to free memory
    public func unloadModel() {
        preprocessorModel = nil
        encoderModel = nil
        decoderJoinerModel = nil
        tokenizer = nil
        isReady = false
    }
    
    // MARK: - Model Loading
    
    /// Load the preprocessor model (audio -> mel features)
    private func loadPreprocessor(config: MLModelConfiguration) async throws {
        guard let modelURL = findModelURL(named: preprocessorModelName) else {
            throw ParakeetError.modelNotFound(preprocessorModelName)
        }
        
        // Load on background thread to prevent UI freeze
        preprocessorModel = try await Task.detached(priority: .userInitiated) {
            try MLModel(contentsOf: modelURL, configuration: config)
        }.value
    }
    
    /// Load the encoder model (mel features -> acoustic embeddings)
    private func loadEncoder(config: MLModelConfiguration) async throws {
        guard let modelURL = findModelURL(named: encoderModelName) else {
            throw ParakeetError.modelNotFound(encoderModelName)
        }
        
        // Load on background thread to prevent UI freeze
        encoderModel = try await Task.detached(priority: .userInitiated) {
            try MLModel(contentsOf: modelURL, configuration: config)
        }.value
    }
    
    /// Load the combined decoder+joiner model (optimized with argmax outputs)
    private func loadDecoderJoiner(config: MLModelConfiguration) async throws {
        guard let modelURL = findModelURL(named: decoderJoinerModelName) else {
            throw ParakeetError.modelNotFound(decoderJoinerModelName)
        }
        
        // Load on background thread to prevent UI freeze
        decoderJoinerModel = try await Task.detached(priority: .userInitiated) {
            try MLModel(contentsOf: modelURL, configuration: config)
        }.value
    }
    
    /// Load tokenizer from Hugging Face
    private func loadTokenizer() async throws {
        do {
            tokenizer = try await AutoTokenizer.from(pretrained: modelName)
        } catch {
            throw ParakeetError.tokenizerLoadFailed(error.localizedDescription)
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
        
        // For development: Try MLModels directory at project root
        let bundleURL = Bundle.main.bundleURL.deletingLastPathComponent().deletingLastPathComponent()
        let modelsPath = bundleURL.appendingPathComponent("ChatMeet/MLModels/\(named).mlpackage")
        if FileManager.default.fileExists(atPath: modelsPath.path) {
            return modelsPath
        }
        
        return nil
    }
    
    // MARK: - Inference
    
    /// Transcribe audio data to text using Parakeet TDT greedy decoding
    /// For audio longer than 30 seconds, use `transcribeLongAudio` instead
    /// - Parameters:
    ///   - audioData: Raw audio data in WAV format (16kHz, mono, PCM)
    ///   - onProgress: Optional callback for real-time token streaming
    /// - Returns: Final transcribed text
    public func transcribe(_ audioData: Data, onProgress: (@Sendable (String) -> Void)? = nil) async throws -> String {
        guard isReady else {
            throw ParakeetError.modelNotLoaded
        }
        
        guard let preprocessor = preprocessorModel,
              let encoder = encoderModel,
              let decoderJoiner = decoderJoinerModel,
              let tokenizer = tokenizer else {
            throw ParakeetError.modelNotLoaded
        }
        
        // 1. Extract PCM samples from WAV data using AudioPreprocessor
        let samples = try AudioPreprocessor.extractPCMSamples(from: audioData)
        
        // Pad/trim to expected length (30 seconds = 480,000 samples at 16kHz)
        let expectedSamples = 480000
        let paddedSamples = AudioPreprocessor.padOrTrim(samples, to: expectedSamples)
        
        // 2. Run preprocessor (PCM -> mel spectrogram)
        let (melFeatures, featureLengths) = try runPreprocessor(preprocessor, audioSamples: paddedSamples)
        
        // Create featureLengths array for encoder
        let lengthArray = try MLMultiArray(shape: [1], dataType: .int32)
        lengthArray[0] = NSNumber(value: featureLengths)
        
        // 3. Run encoder (mel -> acoustic embeddings)
        // The encoder compresses time steps: 3001 -> 376
        let acousticEmbeddings = try runEncoder(encoder, melFeatures: melFeatures, featureLengths: lengthArray) // (B, D, T)
        
        // 4. Run greedy decoding with optimized decoder+joiner
        let tokenIds = try await runGreedyDecoding(
            decoderJoiner: decoderJoiner,
            acousticEmbeddings: acousticEmbeddings,
            onProgress: onProgress,
            tokenizer: tokenizer
        )
        
        // 5. Decode tokens to text
        let transcription = tokenizer.decode(tokens: tokenIds)
        
        // Clean Parakeet artifacts before returning
        return cleanParakeetArtifacts(transcription)
    }
    
    /// Transcribe audio chunk and return tokens with frame positions for streaming
    /// This method exposes token-level data needed for frame-aligned token merging
    /// - Parameters:
    ///   - audioData: Raw audio data in WAV format (16kHz, mono, PCM)
    ///   - onProgress: Optional callback for real-time token streaming
    /// - Returns: Tuple of (tokens: [Int], tokenStartFrames: [Int], confidences: [Float])
    public func transcribeChunkWithTokens(
        _ audioData: Data,
        onProgress: (@Sendable (String) -> Void)? = nil
    ) async throws -> (tokens: [Int], tokenStartFrames: [Int], confidences: [Float]) {
        guard isReady else {
            throw ParakeetError.modelNotLoaded
        }
        
        guard let preprocessor = preprocessorModel,
              let encoder = encoderModel,
              let decoderJoiner = decoderJoinerModel,
              let tokenizer = tokenizer else {
            throw ParakeetError.modelNotLoaded
        }
        
        // 1. Extract PCM samples from WAV data
        let samples = try AudioPreprocessor.extractPCMSamples(from: audioData)
        
        // Pad/trim to expected length (30 seconds = 480,000 samples at 16kHz)
        let expectedSamples = 480000
        let paddedSamples = AudioPreprocessor.padOrTrim(samples, to: expectedSamples)
        
        // 2. Run preprocessor (PCM -> mel spectrogram)
        let (melFeatures, featureLengths) = try runPreprocessor(preprocessor, audioSamples: paddedSamples)
        
        // Create featureLengths array for encoder
        let lengthArray = try MLMultiArray(shape: [1], dataType: .int32)
        lengthArray[0] = NSNumber(value: featureLengths)
        
        // 3. Run encoder (mel -> acoustic embeddings)
        let acousticEmbeddings = try runEncoder(encoder, melFeatures: melFeatures, featureLengths: lengthArray)
        
        // 4. Run greedy decoding with frame tracking
        let (tokenIds, tokenFrames, tokenConfidences) = try await runGreedyDecodingWithFrames(
            decoderJoiner: decoderJoiner,
            acousticEmbeddings: acousticEmbeddings,
            onProgress: onProgress,
            tokenizer: tokenizer
        )
        
        return (tokens: tokenIds, tokenStartFrames: tokenFrames, confidences: tokenConfidences)
    }
    
    // MARK: - Helper Methods

    
    /// Transcribe long audio using rolling window strategy
    /// - Parameters:
    ///   - audioData: Raw audio data in WAV format (16kHz, mono, PCM)
    ///   - windowDuration: Duration of each window in seconds (default: 20s for batch, 5s for streaming)
    ///   - overlapDuration: Overlap between windows in seconds (default: 2s)
    ///   - onProgress: Optional callback for progress updates with partial transcription
    /// - Returns: Complete transcribed text
    public func transcribeLongAudio(
        _ audioData: Data,
        windowDuration: TimeInterval = 20.0,
        overlapDuration: TimeInterval = 2.0,
        onProgress: (@Sendable (String) -> Void)? = nil
    ) async throws -> String {
        guard isReady else {
            throw ParakeetError.modelNotLoaded
        }
        
        // 1. Extract PCM samples
        let allSamples = try AudioPreprocessor.extractPCMSamples(from: audioData)
        let sampleRate = 16000
        let totalDuration = Double(allSamples.count) / Double(sampleRate)
        
        // 2. Calculate window parameters
        let windowSamples = Int(windowDuration * Double(sampleRate))
        let overlapSamples = Int(overlapDuration * Double(sampleRate))
        let hopSamples = windowSamples - overlapSamples
        
        guard windowSamples <= 480000 else {
            throw ParakeetError.inferenceFailed("Window duration too long (max 30s)")
        }
        
        // 3. Process audio in windows
        var transcriptionSegments: [String] = []
        var startSample = 0
        var windowIndex = 0
        
        while startSample < allSamples.count {
            let endSample = min(startSample + windowSamples, allSamples.count)
            let windowSampleSlice = Array(allSamples[startSample..<endSample])
            
            // Pad to expected length for model
            let paddedWindow = AudioPreprocessor.padOrTrim(windowSampleSlice, to: 480000)
            
            // Convert to WAV data for transcription
            let windowWavData = AudioPreprocessor.convertSamplesToWAV(paddedWindow, sampleRate: sampleRate)
            
            // Transcribe this window
            let segmentText = try await transcribe(windowWavData, onProgress: nil)
            
            // For windows after the first, we need to estimate and skip the overlapping portion
            if windowIndex > 0 && !segmentText.isEmpty {
                // Calculate what percentage of this window is overlap vs new
                let overlapRatio = Double(overlapSamples) / Double(windowSampleSlice.count)
                
                // Split transcription: skip roughly the first portion that corresponds to overlap
                let words = segmentText.split(separator: " ").map(String.init)
                let wordsToSkip = Int(Double(words.count) * overlapRatio)
                
                if wordsToSkip < words.count {
                    let newWords = words.dropFirst(wordsToSkip)
                    if !newWords.isEmpty {
                        transcriptionSegments.append(newWords.joined(separator: " "))
                    }
                }
            } else if !segmentText.isEmpty {
                // First window - keep everything
                transcriptionSegments.append(segmentText)
            }
            
            // Report progress
            if let onProgress = onProgress, !transcriptionSegments.isEmpty {
                let combinedText = transcriptionSegments.joined(separator: " ")
                onProgress(combinedText)
            }
            
            // Move to next window
            startSample += hopSamples
            windowIndex += 1
            
            // Safety check to prevent infinite loop
            if hopSamples <= 0 {
                break
            }
        }
        
        // 4. Combine segments (simple concatenation now since overlap is handled)
        let finalText = transcriptionSegments.joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        
        return finalText
    }
    
    /// Deduplicate overlapping text at segment boundaries
    private func deduplicateTranscriptionSegments(_ segments: [String], overlapDuration: TimeInterval) -> String {
        guard segments.count > 1 else {
            return cleanParakeetArtifacts(segments.first ?? "")
        }
        
        var result = segments[0]
        
        for i in 1..<segments.count {
            let currentSegment = segments[i]
            
            // Clean Parakeet artifacts (dots from silence) before processing
            let cleanResult = cleanParakeetArtifacts(result)
            let cleanCurrent = cleanParakeetArtifacts(currentSegment)
            
            // Try to find overlap by checking if the end of result matches beginning of current segment
            // Use word-level matching for better accuracy
            let resultWords = cleanResult.split(separator: " ").map(String.init).filter { !$0.isEmpty }
            let currentWords = cleanCurrent.split(separator: " ").map(String.init).filter { !$0.isEmpty }
            
            // Look for longest common suffix/prefix (up to 50% of either segment)
            var bestOverlapLength = 0
            let maxOverlapWords = min(resultWords.count / 2, currentWords.count / 2, 10)
            
            for overlapLength in (1...maxOverlapWords).reversed() {
                let resultSuffix = resultWords.suffix(overlapLength)
                let currentPrefix = currentWords.prefix(overlapLength)
                
                if Array(resultSuffix) == Array(currentPrefix) {
                    bestOverlapLength = overlapLength
                    break
                }
            }
            
            if bestOverlapLength > 0 {
                // Found overlap, skip those words from current segment
                let remainingWords = currentWords.dropFirst(bestOverlapLength)
                if !remainingWords.isEmpty {
                    result = cleanResult + " " + remainingWords.joined(separator: " ")
                } else {
                    result = cleanResult
                }
            } else {
                // No overlap found, just concatenate
                if !cleanCurrent.isEmpty {
                    result = cleanResult + " " + cleanCurrent
                } else {
                    result = cleanResult
                }
            }
        }
        
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    
    /// Remove Parakeet model artifacts like "..." or "... " from silence/padding
    private func cleanParakeetArtifacts(_ text: String) -> String {
        var cleaned = text
        
        // Remove patterns of dots that indicate silence/no audio
        // Match "..." (3+ dots with optional spaces)
        cleaned = cleaned.replacingOccurrences(of: #"\s*\.{3,}\s*"#, with: " ", options: .regularExpression)
        
        // Remove standalone dots
        cleaned = cleaned.replacingOccurrences(of: #"\s+\.\s+"#, with: " ", options: .regularExpression)
        
        // Clean up multiple spaces
        cleaned = cleaned.replacingOccurrences(of: #"\s{2,}"#, with: " ", options: .regularExpression)
        
        return cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    
    // MARK: - Preprocessing
    
    /// Run preprocessor to convert PCM audio to mel spectrogram
    private func runPreprocessor(_ model: MLModel, audioSamples: [Float]) throws -> (MLMultiArray, Int) {
        // Create input array with shape [1, 480000] using MLShapedArray
        let shapedAudio = MLShapedArray(scalars: audioSamples, shape: [1, audioSamples.count])
        let audioArray = MLMultiArray(shapedAudio)
        
        // Create audio length array with shape [1]
        let lengthArray = try MLMultiArray(shape: [1], dataType: .int32)
        lengthArray[0] = NSNumber(value: audioSamples.count)
        
        // Create input provider
        let inputProvider = try MLDictionaryFeatureProvider(dictionary: [
            "audio_signal": MLFeatureValue(multiArray: audioArray),
            "audio_length": MLFeatureValue(multiArray: lengthArray)
        ])
        
        // Run model
        let outputProvider = try model.prediction(from: inputProvider)
        
        // Extract features output
        guard let features = outputProvider.featureValue(for: "mel")?.multiArrayValue else {
            throw NSError(domain: "ParakeetModel", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to extract preprocessor mel output"])
        }
        
        // Extract featureLengths output
        // Shape is [1], contains the number of mel frames
        guard let featureLengthsArray = outputProvider.featureValue(for: "mel_length")?.multiArrayValue else {
            throw NSError(domain: "ParakeetModel", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to extract preprocessor mel_length output"])
        }
        
        let featureLengths = featureLengthsArray[0].intValue
        
        return (features, featureLengths)
    }
    
    /// Truncate mel features to a maximum length along the time dimension
    /// - Parameters:
    ///   - melFeatures: Input mel features with shape [1, 128, T]
    ///   - toLength: Maximum length for the time dimension
    /// - Returns: Truncated mel features with shape [1, 128, min(T, toLength)]
    private func truncateMelFeatures(_ melFeatures: MLMultiArray, toLength maxLength: Int) throws -> MLMultiArray {
        let shape = melFeatures.shape
        let timeSteps = shape[2].intValue
        
        // If already within limit, return original
        if timeSteps <= maxLength {
            return melFeatures
        }
        
        // Use MLShapedArray slicing for efficient truncation
        let shapedFeatures = MLShapedArray<Float>(melFeatures)
        let truncatedSlice = shapedFeatures[0..., 0..., 0..<maxLength]
        
        return MLMultiArray(truncatedSlice)
    }
    
    // MARK: - Encoder
    
    /// Run encoder model (mel features -> acoustic embeddings)
    /// - Parameters:
    ///   - encoder: Encoder MLModel
    ///   - melFeatures: Mel spectrogram features from preprocessor
    ///   - featureLengths: Feature lengths from preprocessor
    /// - Returns: Acoustic embeddings [T, 1, D]
    private func runEncoder(_ encoder: MLModel, melFeatures: MLMultiArray, featureLengths: MLMultiArray) throws -> MLMultiArray {
        // Run encoder (input: mel, mel_length)
        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
            "mel": MLFeatureValue(multiArray: melFeatures),
            "mel_length": MLFeatureValue(multiArray: featureLengths)
        ])
        
        let output = try encoder.prediction(from: inputFeatures)

        // Extract acoustic embeddings (output: encoder, encoder_length)
        guard let embeddings = output.featureValue(for: "encoder")?.multiArrayValue else {
            throw ParakeetError.inferenceFailed("Failed to extract acoustic embeddings from encoder")
        }
        
        return embeddings
    }

    
    // MARK: - Greedy Decoding
    
    /// Run greedy decoding with optimized decoder+joiner (RNN-T algorithm)
    /// - Parameters:
    ///   - decoderJoiner: Combined decoder+joiner MLModel with argmax outputs
    ///   - acousticEmbeddings: Acoustic embeddings from encoder [B, D, T]
    ///   - onProgress: Optional progress callback
    ///   - tokenizer: Tokenizer for decoding
    /// - Returns: Array of predicted token IDs
    private func runGreedyDecoding(
        decoderJoiner: MLModel,
        acousticEmbeddings: MLMultiArray,
        onProgress: (@Sendable (String) -> Void)?,
        tokenizer: Tokenizer
    ) async throws -> [Int] {
        // Convert MLMultiArray to MLShapedArray for efficient slicing
        let shapedEmbeddings = MLShapedArray<Float>(acousticEmbeddings)
        let timeSteps = shapedEmbeddings.shape[2]  // [B, D, T]
        
        // For Parakeet TDT, blank token ID is 8192 (vocabulary size)
        let blankId = 8192
        let maxSymbols = 10
        let durations = [0, 1, 2, 3, 4]  // Duration tokens at indices 8193-8197
        
        var ySequence: [Int] = []
        var lastToken: Int? = nil  // nil means we need to initialize with blank
        
        // Initialize decoder LSTM states [num_layers, batch_size, hidden_size]
        let numLayers = 2
        let batchSize = 1
        let hiddenSize = 640
        
        // Initialize states to zeros using MLShapedArray (cleaner than looping)
        var stateH = MLMultiArray(MLShapedArray<Float>(repeating: 0.0, shape: [numLayers, batchSize, hiddenSize]))
        var stateC = MLMultiArray(MLShapedArray<Float>(repeating: 0.0, shape: [numLayers, batchSize, hiddenSize]))
        
        var timeIdx = 0
        var iterationCount = 0
        
        // Streaming optimizations
        var lastUpdateTokenCount = 0
        let updateInterval = 10  // Update UI every 10 tokens
        let yieldInterval = 20   // Yield every 20 iterations
        
        while timeIdx < timeSteps {
            iterationCount += 1
            
            // Yield less frequently to reduce overhead (only when streaming)
            if onProgress != nil && iterationCount % yieldInterval == 0 {
                await Task.yield()
            }
            
            // Extract current frame [1, 1024, 1] from [B, D, T] using slice
            let frameSlice = shapedEmbeddings[0..., 0..., timeIdx...timeIdx]  // Extract single time step
            let encoderStep = MLMultiArray(frameSlice)  // Convert back to MLMultiArray for model input
            
            // Inner loop: keep predicting at same time step until blank or max_symbols
            var symbolsAdded = 0
            var needLoop = true
            var skip = 0  // Track last skip value
            
            while needLoop && symbolsAdded < maxSymbols {
                // Determine input token for decoder
                // Use lastToken if we have one, otherwise use blank for initialization
                let inputToken = lastToken ?? blankId
                
                // Run combined decoder+joiner to get token_id and duration (argmax already computed!)
                let (k, durationIdx) = try runDecoderJoiner(
                    decoderJoiner,
                    token: inputToken,
                    encoderStep: encoderStep,
                    stateH: &stateH,
                    stateC: &stateC
                )
                
                skip = durations[durationIdx]
                
                // Check if blank token
                if k == blankId {
                    // Blank predicted: exit inner loop, move to next time step
                    // Only advance if skip > 0, otherwise advance by 1
                    if skip > 0 {
                        timeIdx += skip
                    } else {
                        timeIdx += 1
                    }
                    needLoop = false
                } else {
                    // Non-blank: add to sequence
                    ySequence.append(k)
                    lastToken = k
                    
                    // Stream progress if callback provided - throttled to avoid flooding main thread
                    if let onProgress = onProgress, 
                       ySequence.count - lastUpdateTokenCount >= updateInterval {
                        // Decode synchronously here since we're throttling
                        let partialText = tokenizer.decode(tokens: ySequence)
                        let cleanedText = cleanParakeetArtifacts(partialText)
                        lastUpdateTokenCount = ySequence.count
                        
                        // Dispatch UI update to main thread (no logging in production)
                        Task { @MainActor in
                            onProgress(cleanedText)
                        }
                    }
                    
                    // Increment counters
                    symbolsAdded += 1
                    
                    // Handle time advancement based on skip
                    if skip > 0 {
                        timeIdx += skip
                        needLoop = false  // Exit inner loop after advancing
                    } else {
                        // skip == 0: continue at same frame (emit multiple tokens)
                        needLoop = true
                    }
                }
            }
            
            // If we hit max_symbols, ensure we advance time
            if symbolsAdded == maxSymbols && skip == 0 {
                timeIdx += 1
            }
        }
        
        // Send final update if we have new tokens since last update
        if let onProgress = onProgress, ySequence.count > lastUpdateTokenCount {
            let finalText = tokenizer.decode(tokens: ySequence)
            let cleanedText = cleanParakeetArtifacts(finalText)
            Task { @MainActor in
                onProgress(cleanedText)
            }
        }
        
        return ySequence
    }
    
    /// Run greedy decoding with frame tracking for token merging
    /// - Parameters:
    ///   - decoderJoiner: Combined decoder+joiner MLModel with argmax outputs
    ///   - acousticEmbeddings: Acoustic embeddings from encoder [B, D, T]
    ///   - onProgress: Optional progress callback
    ///   - tokenizer: Tokenizer for decoding
    /// - Returns: Tuple of (tokens, tokenStartFrames, confidences)
    private func runGreedyDecodingWithFrames(
        decoderJoiner: MLModel,
        acousticEmbeddings: MLMultiArray,
        onProgress: (@Sendable (String) -> Void)?,
        tokenizer: Tokenizer
    ) async throws -> (tokens: [Int], tokenStartFrames: [Int], confidences: [Float]) {
        // Convert MLMultiArray to MLShapedArray for efficient slicing
        let shapedEmbeddings = MLShapedArray<Float>(acousticEmbeddings)
        let timeSteps = shapedEmbeddings.shape[2]  // [B, D, T]
        
        let blankId = 8192
        let maxSymbols = 10
        let durations = [0, 1, 2, 3, 4]
        
        var tokens: [Int] = []
        var tokenStartFrames: [Int] = []
        var tokenConfidences: [Float] = []
        var lastToken: Int? = nil
        
        // Initialize decoder LSTM states
        let numLayers = 2
        let batchSize = 1
        let hiddenSize = 640
        
        var stateH = MLMultiArray(MLShapedArray<Float>(repeating: 0.0, shape: [numLayers, batchSize, hiddenSize]))
        var stateC = MLMultiArray(MLShapedArray<Float>(repeating: 0.0, shape: [numLayers, batchSize, hiddenSize]))
        
        var timeIdx = 0
        var iterationCount = 0
        
        // Streaming optimizations
        var lastUpdateTokenCount = 0
        let updateInterval = 10
        let yieldInterval = 20
        
        while timeIdx < timeSteps {
            iterationCount += 1
            
            if onProgress != nil && iterationCount % yieldInterval == 0 {
                await Task.yield()
            }
            
            // Extract current frame
            let frameSlice = shapedEmbeddings[0..., 0..., timeIdx...timeIdx]
            let encoderStep = MLMultiArray(frameSlice)
            
            var symbolsAdded = 0
            var needLoop = true
            var skip = 0
            
            while needLoop && symbolsAdded < maxSymbols {
                let inputToken = lastToken ?? blankId
                
                // Run decoder+joiner and get confidence
                let (k, durationIdx, confidence) = try runDecoderJoinerWithConfidence(
                    decoderJoiner,
                    token: inputToken,
                    encoderStep: encoderStep,
                    stateH: &stateH,
                    stateC: &stateC
                )
                
                skip = durations[durationIdx]
                
                if k == blankId {
                    // Blank predicted: advance time
                    if skip > 0 {
                        timeIdx += skip
                    } else {
                        timeIdx += 1
                    }
                    needLoop = false
                } else {
                    // Non-blank: record token with its frame position
                    tokens.append(k)
                    tokenStartFrames.append(timeIdx)  // Record frame where token was emitted
                    tokenConfidences.append(confidence)
                    lastToken = k
                    
                    // Stream progress if callback provided
                    if let onProgress = onProgress,
                       tokens.count - lastUpdateTokenCount >= updateInterval {
                        let partialText = tokenizer.decode(tokens: tokens)
                        let cleanedText = cleanParakeetArtifacts(partialText)
                        lastUpdateTokenCount = tokens.count
                        
                        Task { @MainActor in
                            onProgress(cleanedText)
                        }
                    }
                    
                    symbolsAdded += 1
                    
                    if skip > 0 {
                        timeIdx += skip
                        needLoop = false
                    } else {
                        needLoop = true
                    }
                }
            }
            
            // If max_symbols hit, advance time
            if symbolsAdded == maxSymbols && skip == 0 {
                timeIdx += 1
            }
        }
        
        // Send final update
        if let onProgress = onProgress, tokens.count > lastUpdateTokenCount {
            let finalText = tokenizer.decode(tokens: tokens)
            let cleanedText = cleanParakeetArtifacts(finalText)
            Task { @MainActor in
                onProgress(cleanedText)
            }
        }
        
        return (tokens, tokenStartFrames, tokenConfidences)
    }
    
    /// Run combined decoder+joiner on single token with state
    /// - Parameters:
    ///   - decoderJoiner: Combined decoder+joiner MLModel  
    ///   - token: Token ID (Int)
    ///   - encoderStep: Single encoder frame [1, 1024, 1]
    ///   - stateH: Hidden state (will be updated)
    ///   - stateC: Cell state (will be updated)
    /// - Returns: Tuple of (token_id, duration_index)
    private func runDecoderJoiner(
        _ decoderJoiner: MLModel,
        token: Int,
        encoderStep: MLMultiArray,
        stateH: inout MLMultiArray,
        stateC: inout MLMultiArray
    ) throws -> (Int, Int) {
        // Create inputs according to metadata.json
        let targets = try MLMultiArray(shape: [1, 1], dataType: .int32)
        targets[[0, 0] as [NSNumber]] = NSNumber(value: token)
        
        let targetLengths = try MLMultiArray(shape: [1], dataType: .int32)
        targetLengths[0] = NSNumber(value: 1)
        
        // DecoderJoiner input: targets, target_lengths, h_in, c_in, encoder_step
        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
            "targets": MLFeatureValue(multiArray: targets),
            "target_lengths": MLFeatureValue(multiArray: targetLengths),
            "h_in": MLFeatureValue(multiArray: stateH),
            "c_in": MLFeatureValue(multiArray: stateC),
            "encoder_step": MLFeatureValue(multiArray: encoderStep)
        ])
        
        let output = try decoderJoiner.prediction(from: inputFeatures)
        
        // DecoderJoiner output: logits, token_id, duration, h_out, c_out
        guard let tokenId = output.featureValue(for: "token_id")?.multiArrayValue,
              let duration = output.featureValue(for: "duration")?.multiArrayValue else {
            throw ParakeetError.inferenceFailed("Failed to extract token_id or duration from decoder+joiner")
        }
        
        // Update states for next iteration
        if let newStateH = output.featureValue(for: "h_out")?.multiArrayValue {
            stateH = newStateH
        }
        if let newStateC = output.featureValue(for: "c_out")?.multiArrayValue {
            stateC = newStateC
        }
        
        // Extract scalar values from shape [1, 1, 1]
        let k = tokenId[[0, 0, 0] as [NSNumber]].intValue
        let durationIdx = duration[[0, 0, 0] as [NSNumber]].intValue
        
        return (k, durationIdx)
    }
    
    /// Run combined decoder+joiner with confidence extraction
    /// - Parameters:
    ///   - decoderJoiner: Combined decoder+joiner MLModel
    ///   - token: Token ID (Int)
    ///   - encoderStep: Single encoder frame [1, 1024, 1]
    ///   - stateH: Hidden state (will be updated)
    ///   - stateC: Cell state (will be updated)
    /// - Returns: Tuple of (token_id, duration_index, confidence)
    private func runDecoderJoinerWithConfidence(
        _ decoderJoiner: MLModel,
        token: Int,
        encoderStep: MLMultiArray,
        stateH: inout MLMultiArray,
        stateC: inout MLMultiArray
    ) throws -> (Int, Int, Float) {
        // Create inputs
        let targets = try MLMultiArray(shape: [1, 1], dataType: .int32)
        targets[[0, 0] as [NSNumber]] = NSNumber(value: token)
        
        let targetLengths = try MLMultiArray(shape: [1], dataType: .int32)
        targetLengths[0] = NSNumber(value: 1)
        
        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
            "targets": MLFeatureValue(multiArray: targets),
            "target_lengths": MLFeatureValue(multiArray: targetLengths),
            "h_in": MLFeatureValue(multiArray: stateH),
            "c_in": MLFeatureValue(multiArray: stateC),
            "encoder_step": MLFeatureValue(multiArray: encoderStep)
        ])
        
        let output = try decoderJoiner.prediction(from: inputFeatures)
        
        // Extract outputs
        guard let tokenId = output.featureValue(for: "token_id")?.multiArrayValue,
              let duration = output.featureValue(for: "duration")?.multiArrayValue,
              let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            throw ParakeetError.inferenceFailed("Failed to extract outputs from decoder+joiner")
        }
        
        // Update states
        if let newStateH = output.featureValue(for: "h_out")?.multiArrayValue {
            stateH = newStateH
        }
        if let newStateC = output.featureValue(for: "c_out")?.multiArrayValue {
            stateC = newStateC
        }
        
        // Extract scalar values
        let k = tokenId[[0, 0, 0] as [NSNumber]].intValue
        let durationIdx = duration[[0, 0, 0] as [NSNumber]].intValue
        
        // Calculate confidence from logits
        // logits shape: [1, 1, vocab_size] where vocab_size = 8198
        // Apply softmax to get probability, use max probability as confidence
        let logitsArray = MLShapedArray<Float>(logits)
        let vocabSize = logitsArray.shape[2]
        
        // Find max logit for numerical stability
        var maxLogit: Float = -Float.infinity
        for i in 0..<vocabSize {
            let logitValue = logitsArray[0, 0, i].scalar ?? 0.0
            if logitValue > maxLogit {
                maxLogit = logitValue
            }
        }
        
        // Compute softmax and find probability of predicted token
        var sumExp: Float = 0.0
        var tokenProb: Float = 0.0
        
        for i in 0..<vocabSize {
            let logitValue = logitsArray[0, 0, i].scalar ?? 0.0
            let expValue = exp(logitValue - maxLogit)
            sumExp += expValue
            
            if i == k {
                tokenProb = expValue
            }
        }
        
        let confidence = tokenProb / sumExp
        
        return (k, durationIdx, confidence)
    }
}

/// Errors that can occur during Parakeet operations
public enum ParakeetError: LocalizedError {
    case modelNotLoaded
    case modelNotFound(String)
    case tokenizerLoadFailed(String)
    case preprocessingFailed
    case inferenceFailed(String)
    case notImplemented(String)
    
    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "Parakeet model is not loaded"
        case .modelNotFound(let name):
            return "Model not found: \(name)"
        case .tokenizerLoadFailed(let reason):
            return "Failed to load tokenizer: \(reason)"
        case .preprocessingFailed:
            return "Failed to preprocess audio"
        case .inferenceFailed(let reason):
            return "Inference failed: \(reason)"
        case .notImplemented(let message):
            return "Not implemented: \(message)"
        }
    }
}
