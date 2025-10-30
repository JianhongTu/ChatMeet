//
//  MeetingAssistantViewModel.swift
//  MeetingAssistant
//
//  ViewModel for managing the meeting assistant UI state
//

import SwiftUI
import Combine

@MainActor
class MeetingAssistantViewModel: ObservableObject {
    @Published public var isRecording: Bool = false
    @Published public var isPlaying: Bool = false
    @Published public var statusMessage: String = "Click 'Start Recording' to begin"
    @Published public var transcription: String = "Transcribed text will appear here..."
    @Published public var summary: String = "Summary bullet points will appear here..."
    @Published public var recordingDuration: TimeInterval = 0
    @Published public var isSummarizing: Bool = false
    
    // Performance statistics
    @Published public var transcriptionTime: TimeInterval = 0
    @Published public var timeToFirstToken: TimeInterval = 0
    @Published public var summarizationTime: TimeInterval = 0
    @Published public var totalProcessingTime: TimeInterval = 0
    @Published public var tokensPerSecond: Double = 0
    
    private let audioRecorder = AudioRecorder()
    private let transcriptionService = TranscriptionService()
    private let summaryService = SummaryService()
    private var recordingTimer: Timer?
    private var playbackTimer: Timer?
    private var processingStartTime: Date?
    private var transcriptionStartTime: Date?
    private var summarizationStartTime: Date?
    private var firstTokenTime: Date?
    private var tokenCount: Int = 0
    
    public init() {}
    
    /// Toggle audio recording on/off
    public func toggleRecording() {
        if isRecording {
            stopRecording()
        } else {
            startRecording()
        }
    }
    
    /// Start audio recording
    private func startRecording() {
        // Request microphone permission first
        audioRecorder.requestPermission { [weak self] granted in
            Task { @MainActor in
                guard let self = self else { return }
                
                if granted {
                    self.isRecording = true
                    self.transcription = ""
                    self.summary = ""
                    self.recordingDuration = 0
                    
                    // Reset statistics
                    self.transcriptionTime = 0
                    self.timeToFirstToken = 0
                    self.summarizationTime = 0
                    self.totalProcessingTime = 0
                    self.tokensPerSecond = 0
                    self.tokenCount = 0
                    
                    // Start recording
                    self.audioRecorder.startRecording()
                    
                    // Start timer to update recording duration
                    self.startRecordingTimer()
                    self.updateRecordingStatus()
                } else {
                    self.statusMessage = "Microphone permission denied. Please enable in Settings."
                    self.isRecording = false
                }
            }
        }
    }
    
    /// Start timer to track recording duration
    private func startRecordingTimer() {
        recordingTimer?.invalidate()
        recordingTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self = self, self.isRecording else { return }
                self.recordingDuration = self.audioRecorder.recordingDuration
                self.updateRecordingStatus()
            }
        }
    }
    
    /// Update status message with recording duration
    private func updateRecordingStatus() {
        let minutes = Int(recordingDuration) / 60
        let seconds = Int(recordingDuration) % 60
        statusMessage = String(format: "Recording... %d:%02d", minutes, seconds)
    }
    
    /// Stop audio recording and process the audio
    private func stopRecording() {
        // Stop the recording timer
        recordingTimer?.invalidate()
        recordingTimer = nil
        
        isRecording = false
        statusMessage = "Processing audio..."
        
        // Stop recording and get audio data
        audioRecorder.stopRecording()
        guard let audioData = audioRecorder.getRecordingData() else {
            statusMessage = "Failed to capture audio"
            transcription = "Error: Could not read recorded audio file"
            return
        }
        
        // Verify we have audio data
        guard !audioData.isEmpty else {
            statusMessage = "No audio recorded"
            transcription = "Please record some audio before stopping"
            return
        }
        
        Task {
            await processAudio(audioData)
        }
    }
    
    /// Process audio through transcription and automatic summarization pipeline
    private func processAudio(_ audioData: Data) async {
        // Clear previous results
        transcription = ""
        summary = ""
        
        // Start overall processing timer
        processingStartTime = Date()
        
        // Transcribe audio with live streaming
        statusMessage = "Transcribing audio..."
        transcriptionStartTime = Date()
        do {
            let transcribedText = try await transcriptionService.transcribe(audioData) { [weak self] partialText in
                // Stream partial transcription to UI in real-time
                guard let self = self else { return }
                Task { @MainActor in
                    self.transcription = partialText
                }
            }
            
            // Calculate transcription time
            if let startTime = transcriptionStartTime {
                transcriptionTime = Date().timeIntervalSince(startTime)
            }
            
            // Update with final transcription
            transcription = transcribedText
            
            // Check if we got transcription
            guard !transcribedText.isEmpty else {
                statusMessage = "No speech detected in audio"
                return
            }
            
            statusMessage = "Transcription complete. Generating summary..."
            
            // Automatically summarize the transcription
            await automaticSummarization()
            
        } catch {
            // Show error in status, keep any partial results
            statusMessage = "Error: \(error.localizedDescription)"
            
            // If transcription failed but we had started, show helpful message
            if transcription.isEmpty {
                transcription = "Transcription failed. Please check:\n• Microphone is working\n• Audio was recorded\n• Whisper models are loaded"
            }
        }
    }
    
    /// Automatically summarize the current transcription text
    private func automaticSummarization() async {
        // Check if there's text to summarize
        guard !transcription.isEmpty else {
            return
        }
        
        isSummarizing = true
        summary = ""
        firstTokenTime = nil
        tokenCount = 0
        
        // Summarize transcription with real-time streaming
        statusMessage = "Generating summary..."
        summarizationStartTime = Date()
        do {
            let summaryText = try await summaryService.summarize(transcription) { [weak self] partialSummary in
                // Update summary in real-time as tokens are generated
                Task { @MainActor in
                    guard let self = self else { return }
                    
                    // Track time to first token
                    if self.firstTokenTime == nil && !partialSummary.isEmpty {
                        self.firstTokenTime = Date()
                        if let startTime = self.summarizationStartTime {
                            self.timeToFirstToken = Date().timeIntervalSince(startTime)
                        }
                    }
                    
                    // Estimate token count (rough approximation: words / 0.75)
                    let wordCount = partialSummary.split(separator: " ").count
                    self.tokenCount = Int(Double(wordCount) / 0.75)
                    
                    self.summary = partialSummary
                }
            }
            
            // Calculate total summarization time
            if let startTime = summarizationStartTime {
                summarizationTime = Date().timeIntervalSince(startTime)
            }
            
            // Calculate total processing time
            if let startTime = processingStartTime {
                totalProcessingTime = Date().timeIntervalSince(startTime)
            }
            
            // Calculate tokens per second
            if summarizationTime > 0 {
                tokensPerSecond = Double(tokenCount) / summarizationTime
            }
            
            // Update final summary text box
            summary = summaryText
            statusMessage = String(format: "Complete! (%.1fs total, %.1f tok/s)", totalProcessingTime, tokensPerSecond)
        } catch {
            statusMessage = "Error generating summary: \(error.localizedDescription)"
            if summary.isEmpty {
                summary = "Failed to generate summary. Please try again."
            }
        }
        
        isSummarizing = false
    }
    
    /// Toggle playback of last recording
    public func togglePlayback() {
        if isPlaying {
            stopPlayback()
        } else {
            startPlayback()
        }
    }
    
    /// Start playing back the last recording
    private func startPlayback() {
        guard audioRecorder.hasRecording else {
            statusMessage = "No recording available to play"
            return
        }
        
        audioRecorder.playLastRecording()
        isPlaying = true
        statusMessage = "Playing recording..."
        
        // Monitor playback status
        playbackTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self = self else { return }
                if !self.audioRecorder.isPlaying {
                    self.stopPlayback()
                }
            }
        }
    }
    
    /// Stop playback
    private func stopPlayback() {
        audioRecorder.stopPlayback()
        isPlaying = false
        playbackTimer?.invalidate()
        playbackTimer = nil
        statusMessage = "Playback stopped"
    }
    
    /// Check if a recording is available for playback
    public var hasRecording: Bool {
        return audioRecorder.hasRecording
    }
    
    /// Process an uploaded audio file
    /// - Parameter fileURL: URL of the uploaded file
    public func processUploadedFile(fileURL: URL) async {
        statusMessage = "Processing uploaded file..."
        transcription = "Converting audio file..."
        summary = ""
        
        // Convert and process the audio file
        guard let audioData = await audioRecorder.processUploadedFile(fileURL: fileURL) else {
            statusMessage = "Failed to process file"
            transcription = "Error: Could not convert audio file. Supported formats: M4A, WAV"
            return
        }
        
        // Verify we have audio data
        guard !audioData.isEmpty else {
            statusMessage = "File is empty"
            transcription = "Error: The audio file appears to be empty"
            return
        }
        
        // Process through transcription pipeline
        await processAudio(audioData)
    }
}
