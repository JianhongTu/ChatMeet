//
//  ParakeetModel.swift
//  MLModels
//
//  Wrapper for Parakeet TDT v3 speech recognition model
//

import Foundation
@preconcurrency import CoreML
@preconcurrency import Tokenizers

/// Manages the Parakeet Core ML model for speech-to-text transcription
/// Parakeet uses a 4-component RNN-T architecture:
/// 1. Preprocessor: Audio -> Mel spectrogram features
/// 2. Encoder: Mel features -> Acoustic embeddings
/// 3. Decoder: Previous tokens -> Linguistic embeddings
/// 4. Joiner: Combines acoustic + linguistic -> Token predictions
class ParakeetModel: @unchecked Sendable {
    
    // Four Core ML models for RNN-T architecture
    private var preprocessorModel: MLModel?
    private var encoderModel: MLModel?
    private var decoderModel: MLModel?
    private var joinerModel: MLModel?
    
    private var tokenizer: Tokenizer?
    
    // Model configuration
    private let modelName = "ToviTu/parakeet-tdt-0.6b-v3-coreml"
    private let preprocessorModelName = "ParakeetPreprocessor"
    private let encoderModelName = "ParakeetEncoder"
    private let decoderModelName = "ParakeetDecoder"
    private let joinerModelName = "ParakeetJoiner"
    
    // Audio configuration
    private let sampleRate: Float = 16000.0
    
    // Model readiness state
    private(set) var isReady: Bool = false
    
    public init() {
        // Models will be loaded lazily via loadModel()
    }
    
    /// Load all Parakeet Core ML models and tokenizer
    /// This should be called after user selects the Parakeet model
    public func loadModel() async throws {
        print("ParakeetModel: 🔄 Starting model loading...")
        
        // Configure Core ML to use all available accelerators (CPU, GPU, Neural Engine)
        let config = MLModelConfiguration()
        config.computeUnits = .all
        
        // Load all four models
        try await loadPreprocessor(config: config)
        try await loadEncoder(config: config)
        try await loadDecoder(config: config)
        try await loadJoiner(config: config)
        
        // Load tokenizer from Hugging Face
        try await loadTokenizer()
        
        isReady = true
        print("ParakeetModel: ✅ All models loaded successfully")
    }
    
    /// Unload all models to free memory
    public func unloadModel() {
        preprocessorModel = nil
        encoderModel = nil
        decoderModel = nil
        joinerModel = nil
        tokenizer = nil
        isReady = false
        print("ParakeetModel: 🗑️ Models unloaded")
    }
    
    // MARK: - Model Loading
    
    /// Load the preprocessor model (audio -> mel features)
    private func loadPreprocessor(config: MLModelConfiguration) async throws {
        guard let modelURL = findModelURL(named: preprocessorModelName) else {
            throw ParakeetError.modelNotFound(preprocessorModelName)
        }
        
        preprocessorModel = try MLModel(contentsOf: modelURL, configuration: config)
        print("ParakeetModel: ✓ Loaded preprocessor model")
    }
    
    /// Load the encoder model (mel features -> acoustic embeddings)
    private func loadEncoder(config: MLModelConfiguration) async throws {
        guard let modelURL = findModelURL(named: encoderModelName) else {
            throw ParakeetError.modelNotFound(encoderModelName)
        }
        
        encoderModel = try MLModel(contentsOf: modelURL, configuration: config)
        print("ParakeetModel: ✓ Loaded encoder model")
    }
    
    /// Load the decoder model (previous tokens -> linguistic embeddings)
    private func loadDecoder(config: MLModelConfiguration) async throws {
        guard let modelURL = findModelURL(named: decoderModelName) else {
            throw ParakeetError.modelNotFound(decoderModelName)
        }
        
        decoderModel = try MLModel(contentsOf: modelURL, configuration: config)
        print("ParakeetModel: ✓ Loaded decoder model")
    }
    
    /// Load the joiner model (combines acoustic + linguistic embeddings)
    private func loadJoiner(config: MLModelConfiguration) async throws {
        guard let modelURL = findModelURL(named: joinerModelName) else {
            throw ParakeetError.modelNotFound(joinerModelName)
        }
        
        joinerModel = try MLModel(contentsOf: modelURL, configuration: config)
        print("ParakeetModel: ✓ Loaded joiner model")
    }
    
    /// Load tokenizer from Hugging Face
    private func loadTokenizer() async throws {
        do {
            tokenizer = try await AutoTokenizer.from(pretrained: modelName)
            print("ParakeetModel: ✓ Loaded tokenizer from \(modelName)")
        } catch {
            print("ParakeetModel: ✗ Failed to load tokenizer: \(error.localizedDescription)")
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
            print("ParakeetModel: ✓ Found model in MLModels/ directory: \(modelsPath.path)")
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
              let decoder = decoderModel,
              let joiner = joinerModel,
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
        print("ParakeetModel: Preprocessor output shape: \(melFeatures.shape), lengths: \(featureLengths)")
        
        // Create featureLengths array for encoder
        let lengthArray = try MLMultiArray(shape: [1], dataType: .int32)
        lengthArray[0] = NSNumber(value: featureLengths)
        
        // 3. Run encoder (mel -> acoustic embeddings)
        // The encoder compresses time steps: 3001 -> 376
        let acousticEmbeddings = try runEncoder(encoder, melFeatures: melFeatures, featureLengths: lengthArray)
        print("ParakeetModel: Encoder output shape: \(acousticEmbeddings.shape)")
        
        // 4. Run greedy decoding with decoder + joiner
        let tokenIds = try await runGreedyDecoding(
            decoder: decoder,
            joiner: joiner,
            acousticEmbeddings: acousticEmbeddings,
            onProgress: onProgress,
            tokenizer: tokenizer
        )
        
                
        // 5. Decode tokens to text
        let transcription = tokenizer.decode(tokens: tokenIds)
        
        return transcription.trimmingCharacters(in: .whitespacesAndNewlines)
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
        
        print("ParakeetModel: Transcribing long audio - duration: \(String(format: "%.1f", totalDuration))s, window: \(windowDuration)s, overlap: \(overlapDuration)s")
        
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
        var previousWindowEndSample = 0  // Track where last window ended
        
        while startSample < allSamples.count {
            let endSample = min(startSample + windowSamples, allSamples.count)
            let windowSampleSlice = Array(allSamples[startSample..<endSample])
            
            // Pad to expected length for model
            let paddedWindow = AudioPreprocessor.padOrTrim(windowSampleSlice, to: 480000)
            
            // Convert to WAV data for transcription
            let windowWavData = AudioPreprocessor.convertSamplesToWAV(paddedWindow, sampleRate: sampleRate)
            
            // Transcribe this window
            print("ParakeetModel: Processing window \(windowIndex + 1) (\(String(format: "%.1f", Double(startSample) / Double(sampleRate)))s - \(String(format: "%.1f", Double(endSample) / Double(sampleRate)))s)")
            
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
                        print("ParakeetModel: Window \(windowIndex + 1) - skipped \(wordsToSkip) overlap words, kept \(newWords.count) new words")
                    }
                } else {
                    print("ParakeetModel: Window \(windowIndex + 1) - all words appear to be overlap, skipping")
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
            previousWindowEndSample = endSample
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
        
        print("ParakeetModel: Long audio transcription complete - \(windowIndex) windows, \(transcriptionSegments.count) segments")
        
        return finalText
    }
    
    /// Deduplicate overlapping text at segment boundaries
    private func deduplicateTranscriptionSegments(_ segments: [String], overlapDuration: TimeInterval) -> String {
        guard segments.count > 1 else {
            return segments.first ?? ""
        }
        
        var result = segments[0]
        
        for i in 1..<segments.count {
            let currentSegment = segments[i]
            
            // Try to find overlap by checking if the end of result matches beginning of current segment
            // Use word-level matching for better accuracy
            let resultWords = result.split(separator: " ").map(String.init)
            let currentWords = currentSegment.split(separator: " ").map(String.init)
            
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
                    result += " " + remainingWords.joined(separator: " ")
                }
            } else {
                // No overlap found, just concatenate
                if !currentSegment.isEmpty {
                    result += " " + currentSegment
                }
            }
        }
        
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    
    // MARK: - Preprocessing
    
    /// Run preprocessor to convert PCM audio to mel spectrogram
    private func runPreprocessor(_ model: MLModel, audioSamples: [Float]) throws -> (MLMultiArray, Int) {
        // Create input array with shape [1, 480000]
        let audioArray = try MLMultiArray(shape: [1, audioSamples.count as NSNumber], dataType: .float32)
        for (i, sample) in audioSamples.enumerated() {
            audioArray[[0, i] as [NSNumber]] = NSNumber(value: sample)
        }
        
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
        print("ParakeetModel: Extracted mel_length = \(featureLengths), mel shape = \(features.shape)")
        
        return (features, featureLengths)
    }
    
    /// Truncate mel features to a maximum length along the time dimension
    /// - Parameters:
    ///   - melFeatures: Input mel features with shape [1, 128, T]
    ///   - toLength: Maximum length for the time dimension
    /// - Returns: Truncated mel features with shape [1, 128, min(T, toLength)]
    private func truncateMelFeatures(_ melFeatures: MLMultiArray, toLength maxLength: Int) throws -> MLMultiArray {
        let shape = melFeatures.shape
        let batchSize = shape[0].intValue
        let melBins = shape[1].intValue
        let timeSteps = shape[2].intValue
        
        // If already within limit, return original
        if timeSteps <= maxLength {
            return melFeatures
        }
        
        // Create truncated array [1, 128, maxLength]
        let truncated = try MLMultiArray(shape: [NSNumber(value: batchSize), NSNumber(value: melBins), NSNumber(value: maxLength)], dataType: .float32)
        
        // Copy data up to maxLength
        for t in 0..<maxLength {
            for m in 0..<melBins {
                let value = melFeatures[[0, m, t] as [NSNumber]].floatValue
                truncated[[0, m, t] as [NSNumber]] = NSNumber(value: value)
            }
        }
        
        return truncated
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

        print("ParakeetModel: Encoder mel_length = \(featureLengths)")
        
        // Extract acoustic embeddings (output: encoder, encoder_length)
        guard let embeddings = output.featureValue(for: "encoder")?.multiArrayValue else {
            throw ParakeetError.inferenceFailed("Failed to extract acoustic embeddings from encoder")
        }
        
        // Transpose from [1, 1024, T] to [T, 1, 1024] for RNN-T decoding
        return try transposeEncoderOutput(embeddings)
    }
    
    /// Transpose encoder output from [1, D, T] to [T, 1, D]
    private func transposeEncoderOutput(_ input: MLMultiArray) throws -> MLMultiArray {
        let shape = input.shape
        let batchSize = shape[0].intValue
        let hiddenDim = shape[1].intValue
        let timeSteps = shape[2].intValue
        
        // Create output array [T, 1, D]
        let output = try MLMultiArray(shape: [NSNumber(value: timeSteps), NSNumber(value: batchSize), NSNumber(value: hiddenDim)], dataType: .float32)
        
        // Transpose
        for t in 0..<timeSteps {
            for d in 0..<hiddenDim {
                let value = input[[0, d, t] as [NSNumber]].floatValue
                output[[t, 0, d] as [NSNumber]] = NSNumber(value: value)
            }
        }
        
        return output
    }
    
    // MARK: - Greedy Decoding
    
    /// Run greedy decoding with decoder and joiner (RNN-T algorithm)
    /// - Parameters:
    ///   - decoder: Decoder MLModel
    ///   - joiner: Joiner MLModel
    ///   - acousticEmbeddings: Acoustic embeddings from encoder [T, 1, D]
    ///   - onProgress: Optional progress callback
    ///   - tokenizer: Tokenizer for decoding
    /// - Returns: Array of predicted token IDs
    private func runGreedyDecoding(
        decoder: MLModel,
        joiner: MLModel,
        acousticEmbeddings: MLMultiArray,
        onProgress: (@Sendable (String) -> Void)?,
        tokenizer: Tokenizer
    ) async throws -> [Int] {
        let timeSteps = acousticEmbeddings.shape[0].intValue
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
        
        var stateH = try MLMultiArray(shape: [NSNumber(value: numLayers), NSNumber(value: batchSize), NSNumber(value: hiddenSize)], dataType: .float32)
        var stateC = try MLMultiArray(shape: [NSNumber(value: numLayers), NSNumber(value: batchSize), NSNumber(value: hiddenSize)], dataType: .float32)
        
        // Initialize states to zeros
        for i in 0..<(numLayers * batchSize * hiddenSize) {
            stateH[i] = 0.0
            stateC[i] = 0.0
        }
        
        var timeIdx = 0
        var iterationCount = 0
        
        print("ParakeetModel: Starting greedy decoding (T=\(timeSteps), blank=\(blankId))")
        
        while timeIdx < timeSteps {
            iterationCount += 1
            if iterationCount % 50 == 0 {
                print("ParakeetModel: Decoding progress - timeIdx: \(timeIdx)/\(timeSteps), tokens: \(ySequence.count)")
            }
            // Extract current frame [1, 1, D] from [T, 1, D]
            let frame = try extractFrame(from: acousticEmbeddings, at: timeIdx)
            
            // Inner loop: keep predicting at same time step until blank or max_symbols
            var symbolsAdded = 0
            var needLoop = true
            var skip = 0  // Track last skip value
            
            while needLoop && symbolsAdded < maxSymbols {
                // Determine input token for decoder
                // Use lastToken if we have one, otherwise use blank for initialization
                let inputToken = lastToken ?? blankId
                
                let lastLabel = try createTokenArray(tokenId: inputToken)
                
                // Run decoder to get linguistic embedding
                let linguisticEmbedding = try runDecoder(decoder, token: lastLabel, stateH: &stateH, stateC: &stateC)
                
                // Run joiner to get logits for this frame
                let allLogits = try runJoinerForFrame(joiner, acoustic: frame, linguistic: linguisticEmbedding)
                
                // Split logits into vocabulary logits and duration logits
                // allLogits shape: [8198] (8192 vocab + 1 blank + 5 duration)
                let vocabSize = 8192
                let vocabLogits = try extractVocabLogits(allLogits, vocabSize: vocabSize, numDurations: durations.count)
                let durationLogits = try extractDurationLogits(allLogits, vocabSize: vocabSize, numDurations: durations.count)
                
                // Get best token from vocabulary (including blank at 8192)
                let k = argmax(vocabLogits)
                
                // Get best duration
                let durationIdx = argmax(durationLogits)
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
                    
                    // Stream progress if callback provided
                    if let onProgress = onProgress {
                        let partialText = tokenizer.decode(tokens: ySequence)
                        onProgress(partialText)
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
        
        print("ParakeetModel: Greedy decoding complete (\(ySequence.count) tokens)")
        
        return ySequence
    }
    
    /// Extract a single frame from acoustic embeddings
    private func extractFrame(from embeddings: MLMultiArray, at timeIdx: Int) throws -> MLMultiArray {
        let hiddenDim = embeddings.shape[2].intValue
        let frame = try MLMultiArray(shape: [1, 1, NSNumber(value: hiddenDim)], dataType: .float32)
        
        for d in 0..<hiddenDim {
            let value = embeddings[[timeIdx, 0, d] as [NSNumber]].floatValue
            frame[[0, 0, d] as [NSNumber]] = NSNumber(value: value)
        }
        
        return frame
    }
    
    /// Create token array [1, 1] from token ID
    private func createTokenArray(tokenId: Int) throws -> MLMultiArray {
        let tokenArray = try MLMultiArray(shape: [1, 1], dataType: .int32)
        tokenArray[[0, 0] as [NSNumber]] = NSNumber(value: tokenId)
        return tokenArray
    }
    
    /// Run decoder on single token with state
    /// - Parameters:
    ///   - decoder: Decoder MLModel
    ///   - token: Token ID array [1, 1]
    ///   - stateH: Hidden state (will be updated)
    ///   - stateC: Cell state (will be updated)
    /// - Returns: Linguistic embedding
    private func runDecoder(_ decoder: MLModel, token: MLMultiArray, stateH: inout MLMultiArray, stateC: inout MLMultiArray) throws -> MLMultiArray {
        // Create target_length array [1]
        let targetLength = try MLMultiArray(shape: [1], dataType: .int32)
        targetLength[0] = NSNumber(value: 1)
        
        // Decoder input: targets, target_length, h_in, c_in
        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
            "targets": MLFeatureValue(multiArray: token),
            "target_length": MLFeatureValue(multiArray: targetLength),
            "h_in": MLFeatureValue(multiArray: stateH),
            "c_in": MLFeatureValue(multiArray: stateC)
        ])
        
        let output = try decoder.prediction(from: inputFeatures)
        
        // Decoder output: decoder, h_out, c_out
        guard let embedding = output.featureValue(for: "decoder")?.multiArrayValue else {
            throw ParakeetError.inferenceFailed("Failed to extract linguistic embedding from decoder")
        }
        
        // Update states for next iteration
        if let newStateH = output.featureValue(for: "h_out")?.multiArrayValue {
            stateH = newStateH
        }
        if let newStateC = output.featureValue(for: "c_out")?.multiArrayValue {
            stateC = newStateC
        }
        
        return embedding
    }
    
    /// Run joiner for a single frame (used in inner decoding loop)
    /// - Parameters:
    ///   - joiner: Joiner MLModel
    ///   - acoustic: Single acoustic frame [1, 1, D]
    ///   - linguistic: Linguistic embedding [1, 640, 1]
    /// - Returns: Full logits array [8198] (vocab + blank + durations)
    private func runJoinerForFrame(_ joiner: MLModel, acoustic: MLMultiArray, linguistic: MLMultiArray) throws -> [Float] {
        // Need to reshape acoustic frame to match encoder input format [1, D, T]
        // acoustic is [1, 1, 1024], need to transpose to [1, 1024, 1]
        let hiddenDim = acoustic.shape[2].intValue
        let acousticReshaped = try MLMultiArray(shape: [1, NSNumber(value: hiddenDim), 1], dataType: .float32)
        
        for d in 0..<hiddenDim {
            let value = acoustic[[0, 0, d] as [NSNumber]].floatValue
            acousticReshaped[[0, d, 0] as [NSNumber]] = NSNumber(value: value)
        }
        
        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
            "encoder": MLFeatureValue(multiArray: acousticReshaped),
            "decoder": MLFeatureValue(multiArray: linguistic)
        ])
        
        let output = try joiner.prediction(from: inputFeatures)
        
        guard let logitsArray = output.featureValue(for: "logits")?.multiArrayValue else {
            throw ParakeetError.inferenceFailed("Failed to extract logits from joiner")
        }
        
        // Extract logits: Shape is [1, 1, 1, 8198], extract [0, 0, 0, :]
        let vocabSize = 8198
        var logits = [Float](repeating: 0, count: vocabSize)
        
        for i in 0..<vocabSize {
            logits[i] = logitsArray[[0, 0, 0, i] as [NSNumber]].floatValue
        }
        
        return logits
    }
    
    /// Extract vocabulary logits (first 8193 elements: 8192 vocab + 1 blank)
    private func extractVocabLogits(_ allLogits: [Float], vocabSize: Int, numDurations: Int) throws -> [Float] {
        let vocabLength = vocabSize + 1  // vocab + blank
        return Array(allLogits[0..<vocabLength])
    }
    
    /// Extract duration logits (last 5 elements) and apply log softmax
    private func extractDurationLogits(_ allLogits: [Float], vocabSize: Int, numDurations: Int) throws -> [Float] {
        let startIdx = vocabSize + 1  // After vocab + blank
        let durationLogits = Array(allLogits[startIdx..<(startIdx + numDurations)])
        
        // Apply log softmax
        let maxLogit = durationLogits.max() ?? 0
        let expSum = durationLogits.reduce(0.0) { $0 + exp($1 - maxLogit) }
        let logSumExp = maxLogit + log(expSum)
        
        return durationLogits.map { $0 - logSumExp }
    }
    
    /// Get index of maximum value (argmax)
    private func argmax(_ array: [Float]) -> Int {
        var maxIndex = 0
        var maxValue = array[0]
        
        for (i, value) in array.enumerated() {
            if value > maxValue {
                maxValue = value
                maxIndex = i
            }
        }
        
        return maxIndex
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
