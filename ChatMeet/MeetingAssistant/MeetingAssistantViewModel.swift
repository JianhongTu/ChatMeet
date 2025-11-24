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
    @Published public var statusMessage: String = "Select a transcription model to begin"
    @Published public var transcription: String = "Transcribed text will appear here..."
    @Published public var summaryBulletPoints: [SummaryBulletPoint] = []
    @Published public var summaryActionLog: String = ""  // Shows LLM decision-making
    @Published public var recordingDuration: TimeInterval = 0
    @Published public var isSummarizing: Bool = false
    @Published public var isStreamingMode: Bool = false  // Toggle for streaming vs batch mode
    @Published public var useOnlineSummarization: Bool = false  // Toggle for online API
    
    // Model selection
    @Published public var selectedModel: TranscriptionModel = .none
    @Published public var isTranscriptionModelReady: Bool = false
    @Published public var isSummarizationModelReady: Bool = false
    @Published public var isLoadingModel: Bool = false
    @Published public var selectedBackend: ComputeBackend = .all
    
    // Performance statistics
    @Published public var transcriptionTime: TimeInterval = 0
    @Published public var timeToFirstToken: TimeInterval = 0
    @Published public var summarizationTime: TimeInterval = 0
    @Published public var totalProcessingTime: TimeInterval = 0
    @Published public var tokensPerSecond: Double = 0
    @Published public var realTimeFactor: Double = 0  // RTF = processing time / audio duration
    
    private let audioRecorder = AudioRecorder()
    private let transcriptionService = TranscriptionService()
    private lazy var summaryService: SummaryService = {
        // Start with online provider if API key is configured, otherwise local
        let provider: SummarizationProvider = APIKeyManager.shared.hasAPIKey() ? .online : .local
        return SummaryService(provider: provider)
    }()
    private var recordingTimer: Timer?
    private var playbackTimer: Timer?
    private var processingStartTime: Date?
    private var transcriptionStartTime: Date?
    private var summarizationStartTime: Date?
    private var firstTokenTime: Date?
    private var tokenCount: Int = 0
    private var streamingBuffer: String = ""  // Accumulates streaming transcriptions
    private var streamingStartTime: Date?  // Track streaming recording start time
    
    public init() {
        // Check if API key is configured and set initial state
        if APIKeyManager.shared.hasAPIKey() {
            self.useOnlineSummarization = true  // Default to online if API key exists
        }
        
        // Check model readiness periodically since they load asynchronously
        Task { @MainActor in
            // Wait a bit for models to start loading
            try? await Task.sleep(nanoseconds: 1_000_000_000) // 1 second
            
            // Poll for summarization model readiness
            for _ in 0..<10 {
                if self.summaryService.isReady {
                    self.isSummarizationModelReady = true
                    break
                }
                try? await Task.sleep(nanoseconds: 500_000_000) // 0.5 seconds
            }
        }
    }
    
    /// Toggle between local and online summarization
    public func toggleSummarizationProvider() {
        if useOnlineSummarization {
            // Switch to online
            if let apiKey = APIKeyManager.shared.getAPIKey() {
                let endpoint = APIKeyManager.shared.getAPIEndpoint()
                let model = APIKeyManager.shared.getModelName()
                summaryService.switchToOnlineProvider(apiKey: apiKey, endpoint: endpoint, model: model)
                statusMessage = "Using online API for summarization"
            } else {
                // No API key configured, keep it off
                useOnlineSummarization = false
                statusMessage = "Please configure API key in settings"
            }
        } else {
            // Switch to local
            summaryService.switchToLocalProvider()
            statusMessage = "Using on-device model for summarization"
        }
    }
    
    /// Switch to a different transcription model
    /// - Parameter model: The model to switch to
    public func switchModel(to model: TranscriptionModel) async {
        // Don't do anything if "none" is selected
        guard model != .none else {
            statusMessage = "Select a transcription model to begin"
            isTranscriptionModelReady = false
            return
        }
        
        guard !isRecording && !isLoadingModel else {
            statusMessage = "Cannot switch model while recording or loading"
            return
        }
        
        isLoadingModel = true
        isTranscriptionModelReady = false
        statusMessage = "Loading \(model.displayName)..."
        
        do {
            try await transcriptionService.switchModel(to: model)
            selectedModel = model
            isTranscriptionModelReady = transcriptionService.isTranscriptionModelReady
            statusMessage = "\(model.displayName) ready! Click 'Start Recording' to begin"
        } catch {
            statusMessage = "Failed to load model: \(error.localizedDescription)"
            isTranscriptionModelReady = false
        }
        
        isLoadingModel = false
    }
    
    /// Switch compute backend for the transcription model
    /// - Parameter backend: The compute backend to use
    public func switchBackend(to backend: ComputeBackend) async {
        guard !isRecording && !isLoadingModel else {
            statusMessage = "Cannot switch backend while recording or loading"
            return
        }
        
        // Only switch if a model is loaded
        guard selectedModel != .none else {
            selectedBackend = backend
            statusMessage = "Backend set to \(backend.description). Load a model to apply."
            return
        }
        
        isLoadingModel = true
        statusMessage = "Switching to \(backend.description) backend..."
        
        do {
            try await transcriptionService.switchBackend(to: backend)
            selectedBackend = backend
            statusMessage = "\(selectedModel.displayName) using \(backend.description) - Ready!"
        } catch {
            statusMessage = "Failed to switch backend: \(error.localizedDescription)"
        }
        
        isLoadingModel = false
    }
    
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
                    self.summaryBulletPoints = []
                    self.summaryActionLog = ""
                    self.recordingDuration = 0
                    self.streamingBuffer = ""
                    self.summaryService.reset()  // Reset summary state
                    
                    // Reset statistics
                    self.transcriptionTime = 0
                    self.timeToFirstToken = 0
                    self.summarizationTime = 0
                    self.totalProcessingTime = 0
                    self.tokensPerSecond = 0
                    self.realTimeFactor = 0
                    self.tokenCount = 0
                    
                    // Start recording based on mode
                    if self.isStreamingMode {
                        self.startStreamingRecording()
                    } else {
                        self.audioRecorder.startRecording()
                    }
                    
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
                
                // Update duration based on mode
                if self.isStreamingMode {
                    // Calculate duration from start time
                    if let startTime = self.streamingStartTime {
                        self.recordingDuration = Date().timeIntervalSince(startTime)
                    }
                } else {
                    // Use audio recorder's duration
                    self.recordingDuration = self.audioRecorder.recordingDuration
                }
                
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
    
    /// Start streaming recording with real-time transcription
    private func startStreamingRecording() {
        statusMessage = "Streaming transcription active..."
        transcriptionStartTime = Date()
        processingStartTime = Date()
        streamingStartTime = Date()  // Track streaming start time
        firstTokenTime = nil  // Reset for tracking TTFT
        
        do {
            try transcriptionService.startStreamingTranscription(
                onUpdate: { [weak self] newTranscription in
                    guard let self = self else { return }
                    Task { @MainActor in
                        // Track time to first token (first non-empty transcription)
                        if self.firstTokenTime == nil && !newTranscription.isEmpty {
                            self.firstTokenTime = Date()
                            if let startTime = self.transcriptionStartTime {
                                self.timeToFirstToken = Date().timeIntervalSince(startTime)
                                print("📊 Time to First Token (Transcription): \(self.timeToFirstToken)s")
                            }
                        }
                        
                        // Update transcription in real-time
                        self.transcription = newTranscription
                        self.streamingBuffer = newTranscription
                    }
                }
            )
        } catch {
            statusMessage = "Failed to start streaming: \(error.localizedDescription)"
            isRecording = false
        }
    }
    
    /// Stop audio recording and process the audio
    private func stopRecording() {
        // Capture recording duration before stopping
        let finalRecordingDuration = recordingDuration
        
        // Stop the recording timer
        recordingTimer?.invalidate()
        recordingTimer = nil
        
        isRecording = false
        
        if isStreamingMode {
            // Stop streaming transcription
            transcriptionService.stopStreamingTranscription()
            
            // Preserve the recording duration
            recordingDuration = finalRecordingDuration
            
            // Calculate final statistics
            if let startTime = transcriptionStartTime {
                transcriptionTime = Date().timeIntervalSince(startTime)
            }
            if let startTime = processingStartTime {
                totalProcessingTime = Date().timeIntervalSince(startTime)
            }
            
            // Calculate Real-Time Factor (RTF = transcription time / audio duration)
            if recordingDuration > 0 {
                realTimeFactor = transcriptionTime / recordingDuration
            }
            
            // Clear streaming start time
            streamingStartTime = nil
            
            statusMessage = "Streaming complete. Ready for next recording."
        } else {
            // Original batch mode
            statusMessage = "Processing audio..."
            
            // Stop recording and get audio data
            audioRecorder.stopRecording()
            
            // Preserve the recording duration (stopRecording resets the recorder's time)
            recordingDuration = finalRecordingDuration
            
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
    }
    
    /// Process audio through transcription and automatic summarization pipeline
    private func processAudio(_ audioData: Data) async {
        // Clear previous results
        transcription = ""
        summaryBulletPoints = []
        summaryActionLog = ""
        summaryService.reset()
        
        // Start overall processing timer
        processingStartTime = Date()
        
        // Transcribe audio with live streaming
        statusMessage = "Transcribing audio..."
        transcriptionStartTime = Date()
        firstTokenTime = nil  // Reset for tracking TTFT
        
        do {
            // Automatically use long audio transcription for files > 30s
            let transcribedText = try await transcriptionService.transcribeLongAudio(
                audioData,
                windowDuration: 20.0,  // 20s windows for better context
                overlapDuration: 2.0    // 2s overlap to avoid missing words at boundaries
            ) { [weak self] partialText in
                // Stream partial transcription to UI in real-time
                guard let self = self else { return }
                Task { @MainActor in
                    // Track time to first token (first non-empty transcription)
                    if self.firstTokenTime == nil && !partialText.isEmpty {
                        self.firstTokenTime = Date()
                        if let startTime = self.transcriptionStartTime {
                            self.timeToFirstToken = Date().timeIntervalSince(startTime)
                            print("📊 Time to First Token (Transcription): \(self.timeToFirstToken)s")
                        }
                    }
                    
                    self.transcription = partialText
                }
            }
            
            // Calculate transcription time
            if let startTime = transcriptionStartTime {
                transcriptionTime = Date().timeIntervalSince(startTime)
            }
            
            // Workaround: Clear and set final transcription to avoid duplication
            // Wait a moment for any pending callbacks to complete, then replace with clean final result
            try? await Task.sleep(nanoseconds: 100_000_000) // 100ms delay
            await MainActor.run {
                self.transcription = transcribedText  // Set final clean text
                print("📝 Final merged text: '\(transcribedText)'")
            }
            
            // Check if we got transcription
            guard !transcribedText.isEmpty else {
                statusMessage = "No speech detected in audio"
                return
            }
            
            // Calculate total processing time
            if let startTime = processingStartTime {
                totalProcessingTime = Date().timeIntervalSince(startTime)
            }
            
            // Calculate Real-Time Factor (RTF = transcription time / audio duration)
            if recordingDuration > 0 {
                realTimeFactor = transcriptionTime / recordingDuration
            }
            
            statusMessage = "Transcription complete. Click 'Summarize' to generate summary."
            
        } catch {
            // Show error in status, keep any partial results
            statusMessage = "Error: \(error.localizedDescription)"
            
            // If transcription failed but we had started, show helpful message
            if transcription.isEmpty {
                transcription = "Transcription failed. Please check:\n• Microphone is working\n• Audio was recorded\n• Models are loaded"
            }
        }
    }
    
    /// Manually trigger summarization (used in both streaming and batch modes)
    public func summarize() async {
        // Check if there's text to summarize
        guard !transcription.isEmpty else {
            statusMessage = "No transcript available to summarize"
            return
        }
        
        isSummarizing = true
        summaryActionLog = ""
        firstTokenTime = nil
        tokenCount = 0
        
        // Use agentic workflow to update summary
        statusMessage = "Generating summary..."
        summarizationStartTime = Date()
        do {
            let bulletPoints = try await summaryService.updateSummary(with: transcription) { [weak self] partialActions in
                // Show LLM decision-making in real-time
                Task { @MainActor in
                    guard let self = self else { return }
                    
                    // Estimate token count
                    let wordCount = partialActions.split(separator: " ").count
                    self.tokenCount = Int(Double(wordCount) / 0.75)
                    
                    // Show action log
                    self.summaryActionLog = partialActions
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
            
            // Update bullet points
            summaryBulletPoints = bulletPoints
            statusMessage = String(format: "Complete! %d bullet points (%.1fs, %.1f tok/s)", bulletPoints.count, totalProcessingTime, tokensPerSecond)
        } catch {
            statusMessage = "Error generating summary: \(error.localizedDescription)"
            if summaryBulletPoints.isEmpty {
                summaryActionLog = "Failed to generate summary. Please try again."
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
    /// - Parameter fileURL: URL of the uploaded file (may be security-scoped)
    public func processUploadedFile(fileURL: URL) async {
        // Start accessing security-scoped resource
        let accessingResource = fileURL.startAccessingSecurityScopedResource()
        
        defer {
            if accessingResource {
                fileURL.stopAccessingSecurityScopedResource()
            }
        }
        
        statusMessage = "Processing uploaded file..."
        transcription = "Converting audio file..."
        summaryBulletPoints = []
        summaryActionLog = ""
        summaryService.reset()
        
        // Reset statistics
        recordingDuration = 0
        transcriptionTime = 0
        timeToFirstToken = 0
        summarizationTime = 0
        totalProcessingTime = 0
        tokensPerSecond = 0
        realTimeFactor = 0
        tokenCount = 0
        
        // Convert and process the audio file
        guard let audioData = await audioRecorder.processUploadedFile(fileURL: fileURL) else {
            statusMessage = "Failed to process file"
            transcription = "Error: Could not convert audio file. Supported formats: M4A, WAV, SPHERE (.sph)"
            return
        }
        
        // Verify we have audio data
        guard !audioData.isEmpty else {
            statusMessage = "File is empty"
            transcription = "Error: The audio file appears to be empty"
            return
        }
        
        // Calculate recording duration from audio data
        // Audio is 16kHz PCM, so duration = samples / sample_rate
        do {
            let samples = try AudioPreprocessor.extractPCMSamples(from: audioData)
            recordingDuration = Double(samples.count) / 16000.0
            print("📊 Uploaded file duration: \(recordingDuration)s (\(samples.count) samples)")
        } catch {
            print("⚠️ Could not calculate duration: \(error)")
            // Continue processing even if duration calculation fails
        }
        
        // Process through transcription pipeline
        await processAudio(audioData)
    }
}
