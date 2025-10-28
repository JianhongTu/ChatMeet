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
            // Header section with title and recording button
            VStack(spacing: 16) {
                Text("Meeting Assistant")
                    .font(.system(size: 28, weight: .semibold))
                    .foregroundColor(.primary)
                
                HStack(spacing: 12) {
                    recordButton
                    playbackButton
                    uploadButton
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
            // Header section with title and buttons stacked vertically
            VStack(spacing: 12) {
                Text("Meeting Assistant")
                    .font(.system(size: 24, weight: .semibold))
                    .foregroundColor(.primary)
                
                // Buttons stacked vertically
                VStack(spacing: 10) {
                    recordButton
                    playbackButton
                    uploadButton
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
        .disabled(viewModel.isPlaying)
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
            Text("Transcription")
                .font(.system(size: 14, weight: .semibold))
                .foregroundColor(.secondary)
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
            Text("Summary")
                .font(.system(size: 14, weight: .semibold))
                .foregroundColor(.secondary)
                .padding(.horizontal, 12)
                .padding(.top, 12)
            
            ZStack {
                #if os(macOS)
                Color(nsColor: .textBackgroundColor)
                #else
                Color(uiColor: .systemBackground)
                #endif
                
                ScrollView {
                    Text(viewModel.summary)
                        .font(.system(size: 13))
                        .frame(maxWidth: .infinity, alignment: .topLeading)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 8)
                        .textSelection(.enabled)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// MARK: - File Importer ViewModifier

extension View {
    func fileImporterModifier(isPresented: Binding<Bool>, viewModel: MeetingAssistantViewModel) -> some View {
        self.fileImporter(
            isPresented: isPresented,
            allowedContentTypes: [.audio],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                guard let selectedURL = urls.first else { return }
                
                // Start accessing security-scoped resource
                guard selectedURL.startAccessingSecurityScopedResource() else {
                    viewModel.statusMessage = "Failed to access file"
                    return
                }
                
                defer {
                    selectedURL.stopAccessingSecurityScopedResource()
                }
                
                // Process the file
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

