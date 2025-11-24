//
//  ContentView.swift
//  MeetingAssistant
//
//  Main UI view for the meeting assistant
//

import SwiftUI
import UniformTypeIdentifiers

struct MeetingAssistantContentView: View {
    @StateObject private var viewModel = MeetingAssistantViewModel()
    @State private var isFilePickerPresented = false
    @State private var isAPISettingsPresented = false
    @State private var isPerformanceStatsPresented = false
    
    init() {}
    
    public var body: some View {
        #if os(macOS)
        macOSLayout
        #else
        iOSLayout
        #endif
    }
    
    // MARK: - macOS Layout
    
    private var macOSLayout: some View {
        VStack(spacing: 0) {
            // Header section with controls
            VStack(spacing: 16) {
                // Model selection and status indicators
                HStack(spacing: 20) {
                    modelPicker
                    backendPicker
                    Spacer()
                    modelStatusIndicators
                }
                .padding(.horizontal, 20)
                
                HStack(spacing: 30) {
                    streamingModeToggle
                    summarizationProviderToggle
                }
                
                HStack(spacing: 12) {
                    recordButton
                    playbackButton
                    uploadButton
                    apiSettingsButton
                    performanceStatsButton
                }
                
                statusMessage
            }
            .padding(.vertical, 24)
            .padding(.horizontal, 20)
            
            Divider()
            
            // Content area with transcription and summary text boxes side by side
            HStack(spacing: 0) {
                transcriptionBox
                Divider()
                summaryBox
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .frame(minWidth: 800, minHeight: 500)
        .fileImporterModifier(isPresented: $isFilePickerPresented, viewModel: viewModel)
    }
    
    // MARK: - iOS Layout
    
    private var iOSLayout: some View {
        VStack(spacing: 0) {
            // Header section with controls
            VStack(spacing: 12) {
                // Model selection and status
                VStack(spacing: 8) {
                    modelPicker
                    backendPicker
                    modelStatusIndicators
                }
                .padding(.horizontal, 20)
                
                // Toggles
                VStack(spacing: 8) {
                    streamingModeToggle
                    summarizationProviderToggle
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 8)
                
                // Buttons stacked vertically
                VStack(spacing: 10) {
                    recordButton
                    playbackButton
                    uploadButton
                    apiSettingsButton
                    performanceStatsButton
                }
                .padding(.horizontal, 20)
                
                statusMessage
            }
            .padding(.vertical, 16)
            
            Divider()
            
            // Content area with transcription and summary text boxes stacked vertically
            VStack(spacing: 0) {
                transcriptionBox
                Divider()
                summaryBox
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .fileImporterModifier(isPresented: $isFilePickerPresented, viewModel: viewModel)
    }
    
    // MARK: - Reusable Components
    
    private var modelPicker: some View {
        HStack(spacing: 8) {
            Image(systemName: "brain")
                .font(.system(size: 16))
                .foregroundColor(.accentColor)
            
            Picker("Transcription Model", selection: $viewModel.selectedModel) {
                ForEach(TranscriptionModel.allCases) { model in
                    Text(model.displayName).tag(model)
                }
            }
            .frame(width: 250)
            .disabled(viewModel.isRecording || viewModel.isPlaying || viewModel.isLoadingModel)
            .onChange(of: viewModel.selectedModel) {
                Task {
                    await viewModel.switchModel(to: viewModel.selectedModel)
                }
            }
            
            // Always reserve space for the progress indicator to prevent layout shifts
            ProgressView()
                .scaleEffect(0.7)
                .opacity(viewModel.isLoadingModel ? 1.0 : 0.0)
        }
    }
    
    private var backendPicker: some View {
        HStack(spacing: 8) {
            Image(systemName: "cpu")
                .font(.system(size: 16))
                .foregroundColor(.accentColor)
            
            Picker("Compute Backend", selection: $viewModel.selectedBackend) {
                ForEach(ComputeBackend.allCases, id: \.self) { backend in
                    Text(backend.description).tag(backend)
                }
            }
            .frame(width: 220)
            .disabled(viewModel.isRecording || viewModel.isPlaying || viewModel.isLoadingModel)
            .onChange(of: viewModel.selectedBackend) {
                Task {
                    await viewModel.switchBackend(to: viewModel.selectedBackend)
                }
            }
        }
    }
    
    private var modelStatusIndicators: some View {
        HStack(spacing: 16) {
            HStack(spacing: 6) {
                Circle()
                    .fill(viewModel.isTranscriptionModelReady ? Color.green : Color.gray)
                    .frame(width: 10, height: 10)
                Text("Transcription")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            HStack(spacing: 6) {
                Circle()
                    .fill(viewModel.isSummarizationModelReady ? Color.green : Color.gray)
                    .frame(width: 10, height: 10)
                Text("Summarization")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
    }
    
    private var streamingModeToggle: some View {
        Toggle(isOn: $viewModel.isStreamingMode) {
            HStack(spacing: 6) {
                Image(systemName: "waveform.circle.fill")
                    .font(.system(size: 14))
                Text("Streaming")
                    .font(.system(size: 13))
            }
        }
        .disabled(viewModel.isRecording || viewModel.isPlaying)
        #if os(macOS)
        .toggleStyle(.switch)
        #endif
    }
    
    private var summarizationProviderToggle: some View {
        Toggle(isOn: $viewModel.useOnlineSummarization) {
            HStack(spacing: 6) {
                Image(systemName: viewModel.useOnlineSummarization ? "cloud.fill" : "cpu")
                    .font(.system(size: 14))
                Text("API LM")
                    .font(.system(size: 13))
            }
        }
        .disabled(viewModel.isRecording || viewModel.isPlaying || !APIKeyManager.shared.hasAPIKey())
        .onChange(of: viewModel.useOnlineSummarization) {
            viewModel.toggleSummarizationProvider()
        }
        #if os(macOS)
        .toggleStyle(.switch)
        #endif
    }
    
    private var apiSettingsButton: some View {
        Button(action: {
            isAPISettingsPresented = true
        }) {
            HStack(spacing: 8) {
                Image(systemName: "key.fill")
                    .font(.system(size: 16))
                Text("API Settings")
                    .font(.system(size: 14, weight: .medium))
            }
            #if os(macOS)
            .frame(width: 140, height: 44)
            #else
            .frame(maxWidth: .infinity)
            .frame(height: 50)
            #endif
            .background(Color.gray.opacity(0.2))
            .foregroundColor(.primary)
            .cornerRadius(8)
        }
        .buttonStyle(.plain)
        .sheet(isPresented: $isAPISettingsPresented) {
            APISettingsView()
        }
    }
    
    private var performanceStatsButton: some View {
        Button(action: {
            isPerformanceStatsPresented = true
        }) {
            HStack(spacing: 8) {
                Image(systemName: "chart.bar.fill")
                    .font(.system(size: 16))
                Text("Performance")
                    .font(.system(size: 14, weight: .medium))
            }
            #if os(macOS)
            .frame(width: 140, height: 44)
            #else
            .frame(maxWidth: .infinity)
            .frame(height: 50)
            #endif
            .background(Color.blue.opacity(0.15))
            .foregroundColor(.blue)
            .cornerRadius(8)
        }
        .buttonStyle(.plain)
        .disabled(viewModel.totalProcessingTime <= 0)
        .opacity(viewModel.totalProcessingTime > 0 ? 1.0 : 0.5)
        .sheet(isPresented: $isPerformanceStatsPresented) {
            PerformanceStatsView(
                recordingDuration: viewModel.recordingDuration,
                transcriptionTime: viewModel.transcriptionTime,
                timeToFirstToken: viewModel.timeToFirstToken,
                summarizationTime: viewModel.summarizationTime,
                tokensPerSecond: viewModel.tokensPerSecond,
                realTimeFactor: viewModel.realTimeFactor,
                totalProcessingTime: viewModel.totalProcessingTime
            )
        }
    }
    
    private var recordButton: some View {
        Button(action: {
            viewModel.toggleRecording()
        }) {
            HStack(spacing: 12) {
                Image(systemName: viewModel.isRecording ? "stop.circle.fill" : "mic.circle.fill")
                    .font(.system(size: 24))
                Text(viewModel.isRecording ? "Stop Recording" : "Start Recording")
                    .font(.system(size: 16, weight: .medium))
            }
            #if os(macOS)
            .frame(width: 200, height: 44)
            #else
            .frame(maxWidth: .infinity)
            .frame(height: 50)
            #endif
            .background(viewModel.isRecording ? Color.red : Color.blue)
            .foregroundColor(.white)
            .cornerRadius(8)
        }
        .buttonStyle(.plain)
        .disabled(viewModel.isPlaying || !viewModel.isTranscriptionModelReady)
        .opacity((viewModel.isTranscriptionModelReady && !viewModel.isPlaying) ? 1.0 : 0.5)
    }
    
    private var playbackButton: some View {
        Button(action: {
            viewModel.togglePlayback()
        }) {
            HStack(spacing: 12) {
                Image(systemName: viewModel.isPlaying ? "stop.fill" : "play.fill")
                    .font(.system(size: 20))
                Text(viewModel.isPlaying ? "Stop" : "Play Recording")
                    .font(.system(size: 16, weight: .medium))
            }
            #if os(macOS)
            .frame(width: 180, height: 44)
            #else
            .frame(maxWidth: .infinity)
            .frame(height: 50)
            #endif
            .background(viewModel.isPlaying ? Color.orange : Color.green)
            .foregroundColor(.white)
            .cornerRadius(8)
        }
        .buttonStyle(.plain)
        .disabled(!viewModel.hasRecording || viewModel.isRecording)
        .opacity((viewModel.hasRecording && !viewModel.isRecording) ? 1.0 : 0.5)
    }
    
    private var uploadButton: some View {
        Button(action: {
            isFilePickerPresented = true
        }) {
            HStack(spacing: 12) {
                Image(systemName: "doc.badge.plus")
                    .font(.system(size: 20))
                Text("Upload Audio")
                    .font(.system(size: 16, weight: .medium))
            }
            #if os(macOS)
            .frame(width: 180, height: 44)
            #else
            .frame(maxWidth: .infinity)
            .frame(height: 50)
            #endif
            .background(Color.purple)
            .foregroundColor(.white)
            .cornerRadius(8)
        }
        .buttonStyle(.plain)
        .disabled(viewModel.isRecording || viewModel.isPlaying)
        .opacity((!viewModel.isRecording && !viewModel.isPlaying) ? 1.0 : 0.5)
    }
    
    private var statusMessage: some View {
        Text(viewModel.statusMessage)
            .font(.system(size: 13))
            .foregroundColor(.secondary)
            .frame(height: 20)
    }
    
    private var transcriptionBox: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .center, spacing: 8) {
                Text("Transcription")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundColor(.secondary)
                Spacer()
            }
            .frame(height: 24)
            .padding(.horizontal, 12)
            .padding(.top, 12)
            
            ZStack {
                #if os(macOS)
                Color(nsColor: .textBackgroundColor)
                #else
                Color(uiColor: .systemBackground)
                #endif
                
                TextEditor(text: $viewModel.transcription)
                    .font(.system(size: 13))
                    .scrollContentBackground(.hidden)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 8)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
    
    private var summaryBox: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .center, spacing: 8) {
                Text("Summary (\(viewModel.summaryBulletPoints.count) points)")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundColor(.secondary)
                
                if viewModel.isSummarizing {
                    ProgressView()
                        .scaleEffect(0.7)
                }
                
                Spacer()
                
                // Manual summarization button (shown in both modes)
                if !viewModel.transcription.isEmpty {
                    Button(action: {
                        Task {
                            await viewModel.summarize()
                        }
                    }) {
                        HStack(spacing: 4) {
                            Image(systemName: "sparkles")
                                .font(.system(size: 11))
                            Text("Summarize")
                                .font(.system(size: 11, weight: .medium))
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5)
                        .background(Color.accentColor)
                        .foregroundColor(.white)
                        .cornerRadius(5)
                    }
                    .buttonStyle(.plain)
                    .disabled(viewModel.isSummarizing)
                    .opacity(viewModel.isSummarizing ? 0.5 : 1.0)
                    .transition(.opacity)
                }
            }
            .frame(height: 24)
            .padding(.horizontal, 12)
            .padding(.top, 12)
            .animation(.easeInOut(duration: 0.2), value: viewModel.isStreamingMode)
            
            ZStack {
                #if os(macOS)
                Color(nsColor: .textBackgroundColor)
                #else
                Color(uiColor: .systemBackground)
                #endif
                
                if viewModel.summaryBulletPoints.isEmpty {
                    // Show placeholder or action log
                    ScrollView {
                        VStack(alignment: .leading, spacing: 8) {
                            if !viewModel.summaryActionLog.isEmpty {
                                Text("LLM Decision Process:")
                                    .font(.system(size: 11, weight: .semibold))
                                    .foregroundColor(.secondary)
                                Text(viewModel.summaryActionLog)
                                    .font(.system(size: 11, design: .monospaced))
                                    .foregroundColor(.secondary)
                            } else {
                                Text("Summary bullet points will appear here...")
                                    .font(.system(size: 13))
                                    .foregroundColor(.secondary)
                            }
                        }
                        .frame(maxWidth: .infinity, alignment: .topLeading)
                        .padding(12)
                    }
                } else {
                    // Show bullet points list
                    ScrollView {
                        VStack(alignment: .leading, spacing: 12) {
                            ForEach(viewModel.summaryBulletPoints) { bullet in
                                bulletPointRow(bullet: bullet)
                            }
                        }
                        .padding(12)
                    }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
    
    /// Individual bullet point row
    private func bulletPointRow(bullet: SummaryBulletPoint) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Text("•")
                .font(.system(size: 16, weight: .bold))
                .foregroundColor(.accentColor)
                .padding(.top, 2)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(bullet.content)
                    .font(.system(size: 13))
                    .textSelection(.enabled)
                
                Text("Updated: \(formatTimestamp(bullet.updatedAt))")
                    .font(.system(size: 10))
                    .foregroundColor(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.vertical, 4)
        .padding(.horizontal, 8)
        .background(Color.secondary.opacity(0.05))
        .cornerRadius(6)
        .transition(.opacity.combined(with: .move(edge: .top)))
        .animation(.easeInOut(duration: 0.3), value: bullet.updatedAt)
    }
    
    /// Format timestamp for display
    private func formatTimestamp(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }
    
    // MARK: - File Importer
}

// MARK: - File Importer ViewModifier

extension View {
    func fileImporterModifier(isPresented: Binding<Bool>, viewModel: MeetingAssistantViewModel) -> some View {
        self.fileImporter(
            isPresented: isPresented,
            allowedContentTypes: [
                .audio,
                UTType(filenameExtension: "sph") ?? .data,  // SPHERE/NIST audio files
                UTType(filenameExtension: "wav") ?? .audio, // Explicitly allow WAV
            ],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                guard let selectedURL = urls.first else { return }
                
                // Pass URL to ViewModel - it will handle security-scoped resource access
                Task {
                    await viewModel.processUploadedFile(fileURL: selectedURL)
                }
                
            case .failure(let error):
                viewModel.statusMessage = "File selection failed: \(error.localizedDescription)"
            }
        }
    }
}

#Preview("Default") {
    MeetingAssistantContentView()
}

