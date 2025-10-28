//
//  TranscriptionService.swift
//  MeetingAssistantCore
//
//  Service for transcribing audio to text using Whisper
//

import Foundation

/// Service that handles audio transcription using the Whisper model
class TranscriptionService: @unchecked Sendable {
    
    private let whisperModel: WhisperModel
    
    public init() {
        self.whisperModel = WhisperModel()
    }
    
    /// Transcribe audio data to text
    /// - Parameter audioData: Raw audio data to transcribe (16kHz WAV format)
    /// - Returns: Transcribed text
    public func transcribe(_ audioData: Data) async throws -> String {
        // Validate audio data
        guard !audioData.isEmpty else {
            throw TranscriptionError.emptyAudioData
        }
        
        // Check minimum audio size (44 byte header + some audio data)
        guard audioData.count > 100 else {
            throw TranscriptionError.audioTooShort
        }
        
        print("TranscriptionService: Starting transcription (\(audioData.count) bytes)")
        
        // Process through Whisper model
        do {
            let transcription = try await whisperModel.transcribe(audioData)
            
            // Validate we got some output
            guard !transcription.isEmpty else {
                throw TranscriptionError.noSpeechDetected
            }
            
            print("TranscriptionService: Transcription complete (\(transcription.count) characters)")
            return transcription
            
        } catch let error as WhisperError {
            // Convert Whisper errors to TranscriptionError
            throw TranscriptionError.transcriptionFailed(error.localizedDescription)
        } catch {
            throw TranscriptionError.transcriptionFailed(error.localizedDescription)
        }
    }
}

/// Errors that can occur during transcription
public enum TranscriptionError: LocalizedError {
    case emptyAudioData
    case audioTooShort
    case noSpeechDetected
    case modelNotLoaded
    case transcriptionFailed(String)
    
    public var errorDescription: String? {
        switch self {
        case .emptyAudioData:
            return "No audio data to transcribe"
        case .audioTooShort:
            return "Audio recording is too short"
        case .noSpeechDetected:
            return "No speech detected in the audio"
        case .modelNotLoaded:
            return "Whisper model is not loaded"
        case .transcriptionFailed(let reason):
            return "Transcription failed: \(reason)"
        }
    }
}
