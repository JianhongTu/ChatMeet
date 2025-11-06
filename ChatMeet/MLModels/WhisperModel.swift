//
//  WhisperModel.swift
//  MLModels
//
//  Wrapper for Whisper speech recognition model
//

import Foundation
@preconcurrency import CoreML
@preconcurrency import Tokenizers

/// Context for streaming transcription to maintain state across chunks
class StreamingTranscriptionContext {    
    // Token management (3-stage: committed, context, current)
    var committedTokens: [Int] = []   // All past transcription (not used in generation)
    var contextTokens: [Int] = []     // Last window's tokens (included in next generation for continuity)
    var currentTokens: [Int] = []     // Current chunk being generated
    
    // Audio management
    var contextAudio: [Float] = []    // Audio from last window (up to 10s worth)
    
    // Statistics
    var totalTokensGenerated: Int = 0
    var chunkCount: Int = 0
    
    // Configuration
    let maxContextDuration: Float = 5.0  // Keep up to 5 seconds of context
    let sampleRate: Float = 16000.0
    
    func reset() {
        committedTokens = []
        contextTokens = []
        currentTokens = []
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
    
    public init() {
        // Initialize the Whisper encoder/decoder models and tokenizer
        loadModels()
    }
    
    /// Load the Core ML Whisper encoder and decoder models
    private func loadModels() {
        // Configure Core ML to use CPU and GPU
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU
        
        // Load encoder model
        // Note: Models should be placed in MLModels/ directory
        do {
            if let encoderURL = findModelURL(named: encoderModelName) {
                encoderModel = try MLModel(contentsOf: encoderURL, configuration: config)
            } else {
                print("WhisperModel: ✗ Encoder model not found (\(encoderModelName))")
            }
        } catch {
            print("WhisperModel: ✗ Failed to load encoder model: \(error.localizedDescription)")
        }
        
        // Load decoder model
        do {
            if let decoderURL = findModelURL(named: decoderModelName) {
                decoderModel = try MLModel(contentsOf: decoderURL, configuration: config)
            } else {
                print("WhisperModel: ✗ Decoder model not found (\(decoderModelName))")
            }
        } catch {
            print("WhisperModel: ✗ Failed to load decoder model: \(error.localizedDescription)")
        }
        
        // Load tokenizer from Hugging Face
        Task { [weak self] in
            guard let self = self else { return }
            do {
                self.tokenizer = try await AutoTokenizer.from(pretrained: self.modelName)
            } catch {
                print("WhisperModel: ✗ Failed to load tokenizer: \(error.localizedDescription)")
            }
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
        // Wait for models to be loaded
        guard let encoder = encoderModel else {
            throw WhisperError.modelNotLoaded
        }
        
        // Extract PCM samples from WAV data
        let samples = try extractPCMSamples(from: audioData)
        
        // Pad/trim to expected length and convert to MLMultiArray
        let paddedSamples = padOrTrim(samples, to: expectedSamples)
        let audioInput = try convertToMLMultiArray(paddedSamples)
        
        // Run encoder to get audio features
        let encoderOutput = try runEncoder(encoder, input: audioInput)
        
        // Run decoder with tokenizer to generate text
        guard let decoder = decoderModel, let tokenizer = tokenizer else {
            throw WhisperError.modelNotLoaded
        }
        
        let transcription = try await runDecoder(decoder, encoderOutput: encoderOutput, tokenizer: tokenizer, onProgress: onProgress)
        
        return transcription
    }
    
    /// Transcribe audio incrementally with 3-stage context management
    /// - Parameters:
    ///   - audioData: Audio data (30s ring buffer window)
    ///   - context: Streaming context with committed/context/current stages
    ///   - onProgress: Optional callback for streaming token updates
    /// - Returns: Only the NEW transcription text from this chunk
    /// - Note: Uses 3-stage strategy:
    ///         - Committed: Past transcriptions (not in generation)
    ///         - Context: Last window's audio+tokens (for continuity, up to 10s)
    ///         - Current: New 5s chunk being transcribed
    public func transcribeIncremental(audioData: Data, context: StreamingTranscriptionContext, onProgress: (@Sendable (String) -> Void)? = nil) async throws -> String {
        guard let encoder = encoderModel, let decoder = decoderModel, let tokenizer = tokenizer else {
            throw WhisperError.modelNotLoaded
        }
        
        context.chunkCount += 1
        print("🔵 WhisperModel: Chunk #\(context.chunkCount) - 3-stage processing")
        
        // Extract PCM samples from WAV data (full 30s ring buffer)
        let samples = try extractPCMSamples(from: audioData)
        
        // STAGE 1: Extract the NEWEST 5 seconds (current chunk)
        let chunkDurationSeconds: Float = 5.0
        let chunkSampleCount = Int(chunkDurationSeconds * context.sampleRate)
        let startIndex = max(0, samples.count - chunkSampleCount)
        let currentChunkAudio = Array(samples[startIndex..<samples.count])
        
        // STAGE 2: Combine context audio (up to 5s) + current chunk (5s)
        // This gives the encoder both recent context and new audio
        let maxContextSamples = Int(context.maxContextDuration * context.sampleRate)
        
        // Trim context audio if needed to fit within limit
        let contextToUse = context.contextAudio.suffix(maxContextSamples)
        
        // Combine: [context audio] + [current chunk audio]
        let combinedAudio = Array(contextToUse) + currentChunkAudio
        
        // Pad to 30 seconds (480,000 samples) - put combined audio at END with leading zeros
        var paddedSamples = [Float](repeating: 0.0, count: expectedSamples)
        let insertPosition = expectedSamples - combinedAudio.count
        for (i, sample) in combinedAudio.enumerated() {
            paddedSamples[insertPosition + i] = sample
        }
        
        print("🔵 WhisperModel: Context audio: \(contextToUse.count) samples (~\(Float(contextToUse.count)/context.sampleRate)s)")
        print("🔵 WhisperModel: Current audio: \(currentChunkAudio.count) samples (~\(Float(currentChunkAudio.count)/context.sampleRate)s)")
        print("🔵 WhisperModel: Combined: \(combinedAudio.count) samples, padded to \(expectedSamples)")
        
        // Convert to MLMultiArray and run encoder on combined audio
        let audioInput = try convertToMLMultiArray(paddedSamples)
        let encoderOutput = try runEncoder(encoder, input: audioInput)
        
        // Run decoder incrementally with context, reusing KV cache
        let newTokens = try await runDecoderIncremental(
            decoder,
            encoderOutput: encoderOutput,
            context: context,
            tokenizer: tokenizer,
            onProgress: onProgress
        )
        
        // STAGE 4: Update context for next chunk
        // Move old context tokens to committed
        context.committedTokens.append(contentsOf: context.contextTokens)
        
        // Current tokens become the new context (remove prompt tokens first)
        let promptLength = 4   // Whisper prompt length
        let actualGeneratedTokens = context.currentTokens.count > promptLength ? 
            Array(context.currentTokens.dropFirst(promptLength)) : []
        context.contextTokens = newTokens
        
        // Update context audio: combine old context + current chunk, keep last 10s
        let maxContextSamplesForUpdate = Int(context.maxContextDuration * context.sampleRate)
        let combinedContextAudio = context.contextAudio + currentChunkAudio
        context.contextAudio = Array(combinedContextAudio.suffix(maxContextSamplesForUpdate))
        
        // Update statistics
        context.totalTokensGenerated += newTokens.count

        // Decode only the NEW tokens to text
        // Don't skip prompt tokens since newTokens are already filtered
        let newTranscription = decodeTokens(newTokens, using: tokenizer, skipPrompt: false)
        
        print("🔵 WhisperModel: Chunk #\(context.chunkCount) complete:")
        print("   - Inputs: \(decodeTokens(context.contextTokens, using: tokenizer))")
        print("   - New tokens generated: \(newTranscription)")
        
        
        
        return newTranscription
    }
    
    /// Extract PCM samples from WAV data
    /// - Parameter audioData: WAV file data
    /// - Returns: Array of normalized float samples (mono, 16kHz)
    private func extractPCMSamples(from audioData: Data) throws -> [Float] {
        // Parse RIFF/WAV header
        guard audioData.count >= 44 else {
            throw WhisperError.preprocessingFailed
        }
        
        // Verify RIFF header
        let riffMagic = audioData.subdata(in: 0..<4)
        guard String(data: riffMagic, encoding: .ascii) == "RIFF" else {
            throw WhisperError.preprocessingFailed
        }
        
        // Verify WAVE format
        let waveMagic = audioData.subdata(in: 8..<12)
        guard String(data: waveMagic, encoding: .ascii) == "WAVE" else {
            throw WhisperError.preprocessingFailed
        }
        
        // Find fmt chunk and data chunk
        var offset = 12
        var fmtChunkOffset: Int?
        var dataChunkOffset: Int?
        var dataChunkSize: Int?
        
        while offset + 8 <= audioData.count {
            let chunkID = audioData.subdata(in: offset..<(offset + 4))
            let chunkSize = audioData.withUnsafeBytes { bytes in
                bytes.load(fromByteOffset: offset + 4, as: UInt32.self)
            }
            
            let chunkIDString = String(data: chunkID, encoding: .ascii) ?? ""
            
            if chunkIDString == "fmt " {
                fmtChunkOffset = offset + 8
            } else if chunkIDString == "data" {
                dataChunkOffset = offset + 8
                dataChunkSize = Int(chunkSize)
                break
            }
            
            offset += 8 + Int(chunkSize)
        }
        
        guard let fmtOffset = fmtChunkOffset,
              let dataOffset = dataChunkOffset,
              let dataSize = dataChunkSize else {
            throw WhisperError.preprocessingFailed
        }
        
        // Parse fmt chunk
        let audioFormat = audioData.withUnsafeBytes { bytes in
            bytes.load(fromByteOffset: fmtOffset, as: UInt16.self)
        }
        let numChannels = Int(audioData.withUnsafeBytes { bytes in
            bytes.load(fromByteOffset: fmtOffset + 2, as: UInt16.self)
        })
        let sampleRateRaw = audioData.withUnsafeBytes { bytes in
            bytes.load(fromByteOffset: fmtOffset + 4, as: UInt32.self)
        }
        let bitsPerSample = Int(audioData.withUnsafeBytes { bytes in
            bytes.load(fromByteOffset: fmtOffset + 14, as: UInt16.self)
        })
        
        // Extract PCM data
        let pcmData = audioData.subdata(in: dataOffset..<(dataOffset + dataSize))
        var samples: [Float] = []
        
        // Decode based on format
        if audioFormat == 1 {  // PCM
            if bitsPerSample == 16 {
                samples = decodePCM16(pcmData, channels: numChannels)
            } else if bitsPerSample == 24 {
                samples = decodePCM24(pcmData, channels: numChannels)
            } else if bitsPerSample == 32 {
                samples = decodePCM32(pcmData, channels: numChannels)
            } else {
                throw WhisperError.preprocessingFailed
            }
        } else if audioFormat == 3 {  // IEEE Float
            samples = decodeFloat32(pcmData, channels: numChannels)
        } else {
            throw WhisperError.preprocessingFailed
        }
        
        // Resample if not 16kHz
        if sampleRateRaw != 16000 {
            samples = resampleTo16kHz(samples, fromRate: Float(sampleRateRaw))
        }
        
        return samples
    }
    
    /// Decode 16-bit PCM and downmix to mono
    private func decodePCM16(_ data: Data, channels: Int) -> [Float] {
        let sampleCount = data.count / 2 / channels
        var monoSamples = [Float](repeating: 0, count: sampleCount)
        
        data.withUnsafeBytes { (buffer: UnsafeRawBufferPointer) in
            let int16Buffer = buffer.bindMemory(to: Int16.self)
            for i in 0..<sampleCount {
                var sum: Float = 0
                for ch in 0..<channels {
                    sum += Float(int16Buffer[i * channels + ch]) / 32768.0
                }
                monoSamples[i] = sum / Float(channels)
            }
        }
        
        return monoSamples
    }
    
    /// Decode 24-bit PCM and downmix to mono
    private func decodePCM24(_ data: Data, channels: Int) -> [Float] {
        let sampleCount = data.count / 3 / channels
        var monoSamples = [Float](repeating: 0, count: sampleCount)
        
        data.withUnsafeBytes { (buffer: UnsafeRawBufferPointer) in
            for i in 0..<sampleCount {
                var sum: Float = 0
                for ch in 0..<channels {
                    let offset = (i * channels + ch) * 3
                    let byte1 = Int32(buffer[offset])
                    let byte2 = Int32(buffer[offset + 1])
                    let byte3 = Int32(buffer[offset + 2])
                    var sample24 = (byte3 << 16) | (byte2 << 8) | byte1
                    if sample24 & 0x800000 != 0 {
                        sample24 |= Int32(bitPattern: 0xFF000000)
                    }
                    sum += Float(sample24) / 8388608.0
                }
                monoSamples[i] = sum / Float(channels)
            }
        }
        
        return monoSamples
    }
    
    /// Decode 32-bit PCM and downmix to mono
    private func decodePCM32(_ data: Data, channels: Int) -> [Float] {
        let sampleCount = data.count / 4 / channels
        var monoSamples = [Float](repeating: 0, count: sampleCount)
        
        data.withUnsafeBytes { (buffer: UnsafeRawBufferPointer) in
            let int32Buffer = buffer.bindMemory(to: Int32.self)
            for i in 0..<sampleCount {
                var sum: Float = 0
                for ch in 0..<channels {
                    sum += Float(int32Buffer[i * channels + ch]) / 2147483648.0
                }
                monoSamples[i] = sum / Float(channels)
            }
        }
        
        return monoSamples
    }
    
    /// Decode 32-bit float PCM and downmix to mono
    private func decodeFloat32(_ data: Data, channels: Int) -> [Float] {
        let sampleCount = data.count / 4 / channels
        var monoSamples = [Float](repeating: 0, count: sampleCount)
        
        data.withUnsafeBytes { (buffer: UnsafeRawBufferPointer) in
            let floatBuffer = buffer.bindMemory(to: Float.self)
            for i in 0..<sampleCount {
                var sum: Float = 0
                for ch in 0..<channels {
                    sum += floatBuffer[i * channels + ch]
                }
                monoSamples[i] = sum / Float(channels)
            }
        }
        
        return monoSamples
    }
    
    /// Simple linear resampling to 16kHz
    private func resampleTo16kHz(_ samples: [Float], fromRate: Float) -> [Float] {
        if fromRate == 16000.0 {
            return samples
        }
        
        let ratio = fromRate / 16000.0
        let outputLength = Int(Float(samples.count) / ratio)
        var resampled = [Float](repeating: 0, count: outputLength)
        
        for i in 0..<outputLength {
            let srcPos = Float(i) * ratio
            let srcIdx = Int(srcPos)
            let frac = srcPos - Float(srcIdx)
            
            if srcIdx + 1 < samples.count {
                resampled[i] = samples[srcIdx] * (1.0 - frac) + samples[srcIdx + 1] * frac
            } else if srcIdx < samples.count {
                resampled[i] = samples[srcIdx]
            }
        }
        
        return resampled
    }
    
    /// Pad or trim audio samples to target length
    private func padOrTrim(_ samples: [Float], to targetLength: Int) -> [Float] {
        if samples.count > targetLength {
            return Array(samples.prefix(targetLength))
        } else if samples.count < targetLength {
            var padded = samples
            padded.append(contentsOf: [Float](repeating: 0.0, count: targetLength - samples.count))
            return padded
        }
        return samples
    }
    
    /// Convert float array to MLMultiArray [1, N]
    private func convertToMLMultiArray(_ samples: [Float]) throws -> MLMultiArray {
        let shape = [1, samples.count] as [NSNumber]
        let mlArray = try MLMultiArray(shape: shape, dataType: .float32)
        
        for (i, sample) in samples.enumerated() {
            mlArray[[0, i] as [NSNumber]] = NSNumber(value: sample)
        }
        
        return mlArray
    }
    
    /// Run encoder model on raw audio waveform
    /// - Parameters:
    ///   - encoder: Encoder MLModel
    ///   - input: Raw audio waveform as MLMultiArray [1, 480000]
    /// - Returns: Encoder output features (encoderHiddenStates)
    private func runEncoder(_ encoder: MLModel, input: MLMultiArray) throws -> MLMultiArray {
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
        
        return encoderOutput
    }
    
    /// Run decoder model to generate text tokens
    /// - Parameters:
    ///   - decoder: Decoder MLModel
    ///   - encoderOutput: Output from encoder
    ///   - tokenizer: Tokenizer for decoding
    /// - Returns: Transcribed text
    private func runDecoder(_ decoder: MLModel, encoderOutput: MLMultiArray, tokenizer: Tokenizer, onProgress: (@Sendable (String) -> Void)? = nil) async throws -> String {
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

        // Autoregressive decoding
        for _ in 0..<maxTokens {
            // Prepare decoder input
            let decoderInput = try prepareDecoderInput(
                tokens: generatedTokens,
                encoderOutput: encoderOutput,
                isPrefill: isPrefill
            )
            
            // Run decoder inference with state
            // The state maintains the KV cache across iterations
            let output = try await decoder.prediction(from: decoderInput, using: state)
            
            // Get next token logits
            guard let logits = extractLogits(from: output) else {
                throw WhisperError.inferenceFailed("Failed to extract logits")
            }
            
            // Get most probable token (greedy decoding)
            let nextToken = argmax(logits)
            
            // Check for end of transcription
            if nextToken == eotTokenId {
                break
            }
            
            generatedTokens.append(nextToken)
            isPrefill = false
            
            // Stream partial transcription if callback is provided
            if let onProgress = onProgress {
                let partialTranscription = decodeTokens(generatedTokens, using: tokenizer)
                onProgress(partialTranscription)
            }
        }
        
        // Decode tokens to text
        let transcription = decodeTokens(generatedTokens, using: tokenizer)
        
        return transcription
    }
    
    /// Run decoder incrementally with KV cache reuse
    /// - Parameters:
    ///   - decoder: Decoder MLModel
    ///   - encoderOutput: Output from encoder (new audio segment)
    ///   - context: Streaming context with existing state
    ///   - tokenizer: Tokenizer for decoding
    ///   - onProgress: Optional callback for streaming
    /// - Returns: Array of newly generated token IDs
    private func runDecoderIncremental(_ decoder: MLModel, encoderOutput: MLMultiArray, context: StreamingTranscriptionContext, tokenizer: Tokenizer, onProgress: (@Sendable (String) -> Void)? = nil) async throws -> [Int] {
        let eotTokenId = 50257  // <|endoftext|>
        let startTokenId = 50258  // <|startoftranscript|>
        let maxTokens = 448
        
        var newTokens: [Int] = []
        
        // Create fresh decoder state for this chunk (matches new encoder output)
        let decoderState = decoder.makeState()
        let promptTokens: [Int] = [startTokenId, 50259, 50359, 50363]
        
        // STAGE 3: Initialize with prompt + context tokens for continuity
        // Context tokens provide transcription history for better coherence
        var initialTokens = promptTokens + context.contextTokens
        context.currentTokens = initialTokens

        print("🔵 WhisperModel: Prefill prompt: \(decodeTokens(initialTokens, using: tokenizer))")

        // // Prefill with prompt + context tokens
        // let prefillInput = try prepareDecoderInput(
        //     tokens: initialTokens,
        //     encoderOutput: encoderOutput,
        //     isPrefill: true
        // )
        // _ = try await decoder.prediction(from: prefillInput, using: decoderState)
        
                
        // Generate tokens for this chunk only (fresh state each time)
        let tokensToGenerate = 50  // Generate ~50 tokens per chunk (roughly 5 seconds of speech)
        print("🔵 WhisperModel: Starting decode for new chunk - will generate up to \(tokensToGenerate) tokens")
        var firstToken = true
        for iteration in 0..<tokensToGenerate {
            // Check if we've hit max tokens
            if context.currentTokens.count >= maxTokens {
                print("⚠️ WhisperModel: Hit max token limit (\(maxTokens)) - stopping generation")
                break
            }
            
            // Prepare input with only the last token (since we're using KV cache)
            let decoderInput = try prepareDecoderInput(
                tokens: context.currentTokens,
                encoderOutput: encoderOutput,
                isPrefill: firstToken // Extension mode with KV cache
            )
            firstToken = false
            
            // Run decoder with existing state (reuses KV cache)
            let output: MLFeatureProvider
            do {
                output = try await decoder.prediction(from: decoderInput, using: decoderState)
            } catch {
                print("❌ WhisperModel: Decoder prediction failed at token \(context.currentTokens.count): \(error)")
                throw WhisperError.inferenceFailed("Decoder failed at \(context.currentTokens.count) tokens: \(error.localizedDescription)")
            }
            
            // Extract logits
            guard let logits = extractLogits(from: output) else {
                print("❌ WhisperModel: Failed to extract logits at token \(context.currentTokens.count)")
                throw WhisperError.inferenceFailed("Failed to extract logits")
            }
            
            // Get next token
            let nextToken = argmax(logits)
            
            // Log if we're getting repetitive tokens (possible sign of KV cache issues)
            if context.currentTokens.count > 100 && iteration % 5 == 0 {
                print("🔵 WhisperModel: Token \(context.currentTokens.count): ID=\(nextToken), topLogit=\(logits[nextToken])")
            }
            
            // Check for end of transcription
            if nextToken == eotTokenId {
                print("🔵 WhisperModel: EOT token detected at iteration \(iteration) - stopping generation")
                break
            }
            
            // Add to current tokens and new tokens
            context.currentTokens.append(nextToken)
            newTokens.append(nextToken)
            
            // Stream if callback provided (show cumulative transcription)
            if let onProgress = onProgress {
                let cumulativeTranscription = decodeTokens(newTokens, using: tokenizer, skipPrompt: false)
                onProgress(cumulativeTranscription)
            }
        }
        
        return newTokens
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
            let tokenArray = try MLMultiArray(shape: [1, NSNumber(value: seqLen)], dataType: .int32)
            for (i, t) in tokens.enumerated() {
                tokenArray[[0, i] as [NSNumber]] = NSNumber(value: t)
            }

            // Causal mask: lower-triangular for self-attention during prefill
            let causalMask = try MLMultiArray(shape: [1, 1, NSNumber(value: seqLen), NSNumber(value: seqLen)], dataType: .float32)
            for i in 0..<seqLen {
                for j in 0..<seqLen {
                    let maskVal: Float = j > i ? -Float.infinity : 0.0
                    causalMask[[0, 0, i, j] as [NSNumber]] = NSNumber(value: maskVal)
                }
            }

            let inputDict: [String: MLFeatureValue] = [
                "inputIds": MLFeatureValue(multiArray: tokenArray),
                "encoderHiddenStates": MLFeatureValue(multiArray: encoderOutput),
                "causalMask": MLFeatureValue(multiArray: causalMask)
            ]
            return try MLDictionaryFeatureProvider(dictionary: inputDict)
        } else {
            // Extension step: feed only the last token
            let lastToken = tokens.last!
            let tokenArray = try MLMultiArray(shape: [1, 1] as [NSNumber], dataType: .int32)
            tokenArray[[0, 0] as [NSNumber]] = NSNumber(value: lastToken)

            // With KV cache, mask can be zeros over full past length
            let pastSeqLen = tokens.count
            let causalMask = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: pastSeqLen)], dataType: .float32)
            for j in 0..<pastSeqLen {
                causalMask[[0, 0, 0, j] as [NSNumber]] = 0
            }

            let inputDict: [String: MLFeatureValue] = [
                "inputIds": MLFeatureValue(multiArray: tokenArray),
                "encoderHiddenStates": MLFeatureValue(multiArray: encoderOutput),
                "causalMask": MLFeatureValue(multiArray: causalMask)
            ]
            return try MLDictionaryFeatureProvider(dictionary: inputDict)
        }
    }
    
    /// Extract logits from decoder output
    /// - Parameter output: Decoder output features
    /// - Returns: Logits array
    private func extractLogits(from output: MLFeatureProvider) -> [Float]? {
        let outputName = "logits"
        guard let logitsArray = output.featureValue(for: outputName)?.multiArrayValue else {
            return nil
        }
        
        // Extract last token's logits
        let vocabSize = logitsArray.shape.last?.intValue ?? 51865  // Whisper vocab size
        var logits = [Float](repeating: 0, count: vocabSize)
        
        for i in 0..<vocabSize {
            logits[i] = logitsArray[[0, logitsArray.shape[1].intValue - 1, i] as [NSNumber]].floatValue
        }
        
        return logits
    }
    
    /// Get index of maximum value (argmax)
    /// - Parameter array: Input array
    /// - Returns: Index of maximum value
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
        
        return decoded.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

/// Errors that can occur during Whisper operations
public enum WhisperError: LocalizedError {
    case modelNotLoaded
    case preprocessingFailed
    case inferenceFailed(String)
    
    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "Whisper model is not loaded"
        case .preprocessingFailed:
            return "Failed to preprocess audio"
        case .inferenceFailed(let reason):
            return "Inference failed: \(reason)"
        }
    }
}
