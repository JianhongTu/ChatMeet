//
//  APISettingsView.swift
//  ChatMeet
//
//  View for configuring API settings for online summarization
//

import SwiftUI

struct APISettingsView: View {
    @State private var apiKey: String = ""
    @State private var endpoint: String = "https://ellm.nrp-nautilus.io/v1/chat/completions"
    @State private var modelName: String = "gemma3"
    @State private var showingSavedAlert = false
    @State private var showingError = false
    @State private var errorMessage = ""
    @State private var isTesting = false
    @State private var testResult: String = ""
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        #if os(macOS)
        content
            .frame(minWidth: 500, minHeight: 600)
        #else
        NavigationStack {
            content
        }
        #endif
    }
    
    private var content: some View {
        Form {
            Section {
                Text("Configure your API key to use online summarization. The key is stored securely in your device's Keychain.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Section(header: Text("API Configuration")) {
                SecureField("API Key", text: $apiKey)
                    .textContentType(.password)
                    #if os(iOS)
                    .autocapitalization(.none)
                    #endif
                    .autocorrectionDisabled()
                
                TextField("API Endpoint", text: $endpoint)
                    .textContentType(.URL)
                    #if os(iOS)
                    .autocapitalization(.none)
                    #endif
                    .autocorrectionDisabled()
                
                TextField("Model Name", text: $modelName)
                    #if os(iOS)
                    .autocapitalization(.none)
                    #endif
                    .autocorrectionDisabled()
            }
            
            Section {
                Button(action: saveCredentials) {
                    HStack {
                        Spacer()
                        Text("Save Credentials")
                            .fontWeight(.semibold)
                        Spacer()
                    }
                }
                .disabled(apiKey.isEmpty)
                
                Button(action: testConnection) {
                    HStack {
                        Spacer()
                        if isTesting {
                            ProgressView()
                                .scaleEffect(0.8)
                                .padding(.trailing, 8)
                        }
                        Text(isTesting ? "Testing..." : "Test Connection")
                            .fontWeight(.semibold)
                        Spacer()
                    }
                }
                .disabled(apiKey.isEmpty || isTesting)
                
                if !testResult.isEmpty {
                    Text(testResult)
                        .font(.caption)
                        .foregroundColor(testResult.contains("✅") ? .green : .red)
                }
                
                if APIKeyManager.shared.hasAPIKey() {
                    Button(role: .destructive, action: clearCredentials) {
                        HStack {
                            Spacer()
                            Text("Clear Stored Credentials")
                            Spacer()
                        }
                    }
                }
            }
            
            Section {
                VStack(alignment: .leading, spacing: 8) {
                    Text("How to get your API key:")
                        .font(.subheadline)
                        .fontWeight(.semibold)
                    
                    Text("1. Contact your API provider")
                    Text("2. Obtain an API key")
                    Text("3. Copy and paste it above")
                    Text("4. Configure the endpoint URL if different")
                    
                    Text("\nCommon Issues:")
                        .font(.subheadline)
                        .fontWeight(.semibold)
                        .padding(.top, 8)
                    
                    Text("• HTTP 403: Invalid API key or insufficient permissions")
                    Text("• Check that your API key is active and has access")
                    Text("• Verify the endpoint URL is correct")
                    
                    Text("\nYour API key is stored securely in your device's Keychain and never leaves your device.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .padding(.top, 4)
                }
                .font(.caption)
            }
        }
        .navigationTitle("API Settings")
        #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
        #endif
        .toolbar {
            ToolbarItem(placement: .cancellationAction) {
                Button("Close") {
                    dismiss()
                }
            }
        }
        .alert("Settings Saved", isPresented: $showingSavedAlert) {
            Button("OK", role: .cancel) {
                dismiss()
            }
        } message: {
            Text("Your API credentials have been saved securely.")
        }
        .alert("Error", isPresented: $showingError) {
            Button("OK", role: .cancel) { }
        } message: {
            Text(errorMessage)
        }
        .onAppear(perform: loadCredentials)
    }
    
    private func loadCredentials() {
        if let storedKey = APIKeyManager.shared.getAPIKey() {
            // Show masked version
            apiKey = String(repeating: "•", count: min(storedKey.count, 20))
        }
        if let storedEndpoint = APIKeyManager.shared.getAPIEndpoint() {
            endpoint = storedEndpoint
        }
        if let storedModel = APIKeyManager.shared.getModelName() {
            modelName = storedModel
        }
    }
    
    private func saveCredentials() {
        // Validate API key format (basic check - not empty)
        if apiKey.isEmpty || apiKey.starts(with: "•") {
            errorMessage = "Please enter a valid API key"
            showingError = true
            return
        }
        
        // Don't save if it's the masked version
        if !apiKey.starts(with: "•") {
            APIKeyManager.shared.saveAPIKey(apiKey)
        }
        
        APIKeyManager.shared.saveAPIEndpoint(endpoint)
        APIKeyManager.shared.saveModelName(modelName)
        
        showingSavedAlert = true
    }
    
    private func testConnection() {
        testResult = ""
        isTesting = true
        
        Task {
            do {
                // Use the current (not masked) API key if available
                let testKey = apiKey.starts(with: "•") ? (APIKeyManager.shared.getAPIKey() ?? apiKey) : apiKey
                
                let provider = OnlineSummarizationProvider(
                    apiKey: testKey,
                    apiEndpoint: endpoint,
                    model: modelName
                )
                
                // Try a simple test request
                let _ = try await provider.summarize("Test connection", onProgress: nil)
                
                await MainActor.run {
                    testResult = "✅ Connection successful!"
                    isTesting = false
                }
            } catch let error as OnlineSummarizationError {
                await MainActor.run {
                    switch error {
                    case .httpError(let code):
                        testResult = "❌ HTTP Error \(code): Check API key and permissions"
                    case .apiError(let message):
                        testResult = "❌ API Error: \(message)"
                    case .invalidEndpoint:
                        testResult = "❌ Invalid endpoint URL"
                    case .invalidResponse:
                        testResult = "❌ Invalid response from server"
                    case .missingAPIKey:
                        testResult = "❌ API key is missing"
                    }
                    isTesting = false
                }
            } catch {
                await MainActor.run {
                    testResult = "❌ Error: \(error.localizedDescription)"
                    isTesting = false
                }
            }
        }
    }
    
    private func clearCredentials() {
        APIKeyManager.shared.clearAll()
        apiKey = ""
        endpoint = "https://ellm.nrp-nautilus.io/v1/chat/completions"
        modelName = "gemma3"
    }
}

#Preview {
    APISettingsView()
}
