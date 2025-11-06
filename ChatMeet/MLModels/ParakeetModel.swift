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
    private let modelName = "mlx-community/parakeet-tdt-0.6b-v3"
    private let preprocessorModelName = "ParakeetPreprocessor"
    private let encoderModelName = "ParakeetEncoderInt4"
    private let decoderModelName = "ParakeetDecoderInt4"
    private let joinerModelName = "ParakeetJoinerInt4"
    
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
        
        // Configure Core ML to use CPU and GPU
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU
        
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
    
    // MARK: - Inference (To be implemented)
    
    /// Transcribe audio data to text
    /// - Parameters:
    ///   - audioData: Raw audio data in WAV format (16kHz, mono, PCM)
    ///   - onProgress: Optional callback for real-time token streaming
    /// - Returns: Final transcribed text
    /// - Note: Inference logic to be implemented
    public func transcribe(_ audioData: Data, onProgress: (@Sendable (String) -> Void)? = nil) async throws -> String {
        guard isReady else {
            throw ParakeetError.modelNotLoaded
        }
        
        // TODO: Implement Parakeet RNN-T inference pipeline
        // 1. Extract PCM samples using AudioPreprocessor
        // 2. Run preprocessor to get mel features
        // 3. Run encoder to get acoustic embeddings
        // 4. Run greedy decoding with decoder + joiner
        // 5. Decode tokens to text
        
        throw ParakeetError.notImplemented("Parakeet inference not yet implemented")
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
