//
//  TranscriptionModelConfig.swift
//  MLModels
//
//  Configuration for available transcription models
//

import Foundation

/// Available transcription models
enum TranscriptionModel: String, CaseIterable, Identifiable {
    case none = "none"
    case whisperTiny = "whisper-tiny"
    case parakeetV3 = "parakeet-v3"
    
    var id: String { rawValue }
    
    /// Display name for UI
    var displayName: String {
        switch self {
        case .none:
            return "Select a model..."
        case .whisperTiny:
            return "Whisper Tiny"
        case .parakeetV3:
            return "Parakeet V3"
        }
    }
    
    /// Model description
    var description: String {
        switch self {
        case .none:
            return "Please select a transcription model"
        case .whisperTiny:
            return "Fast, lightweight model"
        case .parakeetV3:
            return "High-quality RNN-T model"
        }
    }
    
    /// Model file names (without extension)
    var modelFiles: [String] {
        switch self {
        case .none:
            return []
        case .whisperTiny:
            return ["WhisperEncoder_whisper-tiny", "WhisperDecoder_whisper-tiny"]
        case .parakeetV3:
            return ["ParakeetPreprocessor", "ParakeetEncoderInt4", "ParakeetDecoderInt4", "ParakeetJoinerInt4"]
        }
    }
    
    /// Tokenizer name on Hugging Face
    var tokenizerName: String {
        switch self {
        case .none:
            return ""
        case .whisperTiny:
            return "openai/whisper-tiny"
        case .parakeetV3:
            return "mlx-community/parakeet-tdt-0.6b-v3"
        }
    }
}
