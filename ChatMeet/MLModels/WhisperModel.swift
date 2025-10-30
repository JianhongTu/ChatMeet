//
//  WhisperModel.swift
//  MLModels
//
//  Wrapper for Whisper speech recognition model
//

import Foundation
@preconcurrency import CoreML
@preconcurrency import Tokenizers

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
        // Try to find in main bundle resources (compiled .mlmodelc)
        if let url = Bundle.main.url(forResource: named, withExtension: "mlmodelc") {
            return url
        }
        
        // Try to find .mlpackage
        if let url = Bundle.main.url(forResource: named, withExtension: "mlpackage") {
            return url
        }
        
        // Try without extension (sometimes Xcode compiles differently)
        if let url = Bundle.main.url(forResource: named, withExtension: nil) {
            return url
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
    /// - Returns: Decoded text string
    private func decodeTokens(_ tokens: [Int], using tokenizer: Tokenizer) -> String {
        // Remove prompt tokens: <|startoftranscript|> <|en|> <|transcribe|> <|no_timestamps|>
        // The first 4 tokens are the prompt, so skip them
        let promptLength = 4
        let textTokens = tokens.count > promptLength ? Array(tokens.dropFirst(promptLength)) : []
        
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
