//
//  AudioRecorder.swift
//  MeetingAssistantCore
//
//  Handles audio recording functionality
//

import Foundation
@preconcurrency import AVFoundation

/// Manages audio input recording for speech transcription
class AudioRecorder: @unchecked Sendable {
    
    private var audioRecorder: AVAudioRecorder?
    private var audioPlayer: AVAudioPlayer?
    #if os(iOS)
    private var audioSession: AVAudioSession?
    #endif
    private var recordingURL: URL?
    private var lastRecordingURL: URL?
    
    public init() {
        setupAudioSession()
    }
    
    /// Configure the audio session for recording
    private func setupAudioSession() {
        #if os(iOS)
        // Configure AVAudioSession for iOS
        audioSession = AVAudioSession.sharedInstance()
        do {
            try audioSession?.setCategory(.record, mode: .default)
            try audioSession?.setActive(true)
        } catch {
            print("AudioRecorder: Failed to setup audio session: \(error)")
        }
        #elseif os(macOS)
        // macOS doesn't use AVAudioSession
        // Permissions are handled through system preferences
        #endif
    }
    
    /// Start recording audio
    public func startRecording() {
        // Create temporary file URL for recording
        let tempDir = FileManager.default.temporaryDirectory
        recordingURL = tempDir.appendingPathComponent("recording_\(Date().timeIntervalSince1970).wav")
        
        guard let url = recordingURL else {
            print("AudioRecorder: Failed to create recording URL")
            return
        }
        
        // Configure audio settings optimized for Whisper (16kHz, mono, PCM)
        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: 16000.0,  // Whisper expects 16kHz
            AVNumberOfChannelsKey: 1,   // Mono
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsFloatKey: false,
            AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue
        ]
        
        do {
            audioRecorder = try AVAudioRecorder(url: url, settings: settings)
            audioRecorder?.prepareToRecord()
            
            let success = audioRecorder?.record() ?? false
            if success {
                print("AudioRecorder: Started recording to \(url.lastPathComponent)")
            } else {
                print("AudioRecorder: Failed to start recording")
            }
        } catch {
            print("AudioRecorder: Error starting recording: \(error)")
        }
    }
    
    /// Stop recording audio
    public func stopRecording() {
        audioRecorder?.stop()
        audioRecorder = nil
        
        // Save the recording URL for playback
        lastRecordingURL = recordingURL
        
        print("AudioRecorder: Recording stopped")
    }
    
    /// Get the audio data from the last recording
    /// - Returns: Audio data in WAV format, or nil if no recording available
    public func getRecordingData() -> Data? {
        guard let url = recordingURL else {
            print("AudioRecorder: No recording URL available")
            return nil
        }
        
        do {
            let audioData = try Data(contentsOf: url)
            print("AudioRecorder: Successfully read \(audioData.count) bytes from recording")
            return audioData
        } catch {
            print("AudioRecorder: Error reading audio file: \(error)")
            return nil
        }
    }
    
    /// Request microphone permission
    /// - Parameter completion: Callback with permission granted status
    public func requestPermission(completion: @escaping @Sendable (Bool) -> Void) {
        #if os(iOS)
        AVAudioSession.sharedInstance().requestRecordPermission { granted in
            DispatchQueue.main.async {
                completion(granted)
            }
        }
        #elseif os(macOS)
        // macOS: Check microphone permission via AVCaptureDevice
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            completion(true)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .audio) { granted in
                DispatchQueue.main.async {
                    completion(granted)
                }
            }
        case .denied, .restricted:
            completion(false)
        @unknown default:
            completion(false)
        }
        #endif
    }
    
    /// Get the current recording duration in seconds
    public var recordingDuration: TimeInterval {
        return audioRecorder?.currentTime ?? 0
    }
    
    /// Check if currently recording
    public var isRecording: Bool {
        return audioRecorder?.isRecording ?? false
    }
    
    /// Play back the last recording
    public func playLastRecording() {
        guard let url = lastRecordingURL else {
            print("AudioRecorder: No recording available to play")
            return
        }
        
        do {
            #if os(iOS)
            // Configure audio session for playback
            try audioSession?.setCategory(.playback, mode: .default)
            try audioSession?.setActive(true)
            #endif
            
            audioPlayer = try AVAudioPlayer(contentsOf: url)
            audioPlayer?.play()
            print("AudioRecorder: Playing recording from \(url.lastPathComponent)")
        } catch {
            print("AudioRecorder: Error playing recording: \(error)")
        }
    }
    
    /// Stop playback
    public func stopPlayback() {
        audioPlayer?.stop()
        audioPlayer = nil
        
        #if os(iOS)
        // Switch back to record mode
        do {
            try audioSession?.setCategory(.record, mode: .default)
        } catch {
            print("AudioRecorder: Error switching back to record mode: \(error)")
        }
        #endif
        
        print("AudioRecorder: Playback stopped")
    }
    
    /// Check if audio is currently playing
    public var isPlaying: Bool {
        return audioPlayer?.isPlaying ?? false
    }
    
    /// Check if a recording is available for playback
    public var hasRecording: Bool {
        return lastRecordingURL != nil
    }
    
    /// Convert M4A audio file to WAV format (16kHz, mono, PCM)
    /// - Parameter sourceURL: URL of the M4A file
    /// - Returns: Data in WAV format, or nil if conversion failed
    public func convertM4AToWAV(sourceURL: URL) async -> Data? {
        let tempDir = FileManager.default.temporaryDirectory
        let outputURL = tempDir.appendingPathComponent("converted_\(Date().timeIntervalSince1970).wav")
        
        do {
            // Read the source audio file
            let audioFile = try AVAudioFile(forReading: sourceURL)
            let format = audioFile.processingFormat
            
            // Create target format: 16kHz, mono, PCM16
            guard let targetFormat = AVAudioFormat(
                commonFormat: .pcmFormatInt16,
                sampleRate: 16000,
                channels: 1,
                interleaved: true
            ) else {
                print("AudioRecorder: Failed to create target format")
                return nil
            }
            
            // Create converter
            guard let converter = AVAudioConverter(from: format, to: targetFormat) else {
                print("AudioRecorder: Failed to create audio converter")
                return nil
            }
            
            // Create output file
            let outputFile = try AVAudioFile(
                forWriting: outputURL,
                settings: targetFormat.settings,
                commonFormat: targetFormat.commonFormat,
                interleaved: true
            )
            
            // Calculate buffer sizes
            let inputFrameCapacity = AVAudioFrameCount(4096)
            guard let inputBuffer = AVAudioPCMBuffer(
                pcmFormat: format,
                frameCapacity: inputFrameCapacity
            ) else {
                print("AudioRecorder: Failed to create input buffer")
                return nil
            }
            
            // Convert and write audio data
            while audioFile.framePosition < audioFile.length {
                let framesToRead = min(inputFrameCapacity, AVAudioFrameCount(audioFile.length - audioFile.framePosition))
                inputBuffer.frameLength = 0
                
                try audioFile.read(into: inputBuffer, frameCount: framesToRead)
                
                // Create output buffer
                let outputFrameCapacity = AVAudioFrameCount(
                    Double(inputBuffer.frameLength) * targetFormat.sampleRate / format.sampleRate
                ) + 1
                
                guard let outputBuffer = AVAudioPCMBuffer(
                    pcmFormat: targetFormat,
                    frameCapacity: outputFrameCapacity
                ) else {
                    print("AudioRecorder: Failed to create output buffer")
                    return nil
                }
                
                var error: NSError?
                nonisolated(unsafe) let buffer = inputBuffer
                let inputBlock: AVAudioConverterInputBlock = { inNumPackets, outStatus in
                    outStatus.pointee = .haveData
                    return buffer
                }
                
                let status = converter.convert(to: outputBuffer, error: &error, withInputFrom: inputBlock)
                
                if let error = error {
                    print("AudioRecorder: Conversion error: \(error)")
                    return nil
                }
                
                if status == .error {
                    print("AudioRecorder: Conversion failed")
                    return nil
                }
                
                if outputBuffer.frameLength > 0 {
                    try outputFile.write(from: outputBuffer)
                }
                
                if status == .endOfStream {
                    break
                }
            }
            
            // Read the converted WAV file
            let wavData = try Data(contentsOf: outputURL)
            
            // Clean up temporary file
            try? FileManager.default.removeItem(at: outputURL)
            
            // Save for playback
            lastRecordingURL = outputURL
            
            print("AudioRecorder: Successfully converted M4A to WAV (\(wavData.count) bytes)")
            return wavData
            
        } catch {
            print("AudioRecorder: Error converting M4A to WAV: \(error)")
            try? FileManager.default.removeItem(at: outputURL)
            return nil
        }
    }
    
    /// Process an uploaded audio file (M4A format)
    /// - Parameter fileURL: URL of the uploaded audio file
    /// - Returns: Audio data in WAV format, or nil if processing failed
    public func processUploadedFile(fileURL: URL) async -> Data? {
        print("AudioRecorder: Processing uploaded file: \(fileURL.lastPathComponent)")
        
        // Check file extension
        let fileExtension = fileURL.pathExtension.lowercased()
        
        if fileExtension == "m4a" {
            // Convert M4A to WAV
            return await convertM4AToWAV(sourceURL: fileURL)
        } else if fileExtension == "wav" {
            // Already WAV, just read it
            do {
                let data = try Data(contentsOf: fileURL)
                lastRecordingURL = fileURL
                print("AudioRecorder: Read WAV file (\(data.count) bytes)")
                return data
            } catch {
                print("AudioRecorder: Error reading WAV file: \(error)")
                return nil
            }
        } else {
            print("AudioRecorder: Unsupported file format: \(fileExtension)")
            return nil
        }
    }
}
