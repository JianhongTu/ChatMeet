import Foundation

/// Manages access to configuration values from xcconfig files
enum AppConfig {
    /// Hugging Face API token for accessing models
    static var huggingFaceToken: String? {
        Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String
    }
    
    /// Check if HF token is configured
    static var isHFTokenConfigured: Bool {
        guard let token = huggingFaceToken else { return false }
        return !token.isEmpty && token != "your_huggingface_token_here"
    }
}
