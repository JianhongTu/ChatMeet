//
//  AudioPreprocessor.swift
//  ChatMeet
//
//  Audio preprocessing using AVFoundation APIs
//  Supports all formats that AVAudioFile can handle (WAV, SPHERE, M4A, MP3, etc.)
//

import Foundation
import AVFoundation

/// Audio preprocessing errors
enum AudioPreprocessingError: Error, LocalizedError {
    case invalidFormat
    case emptyData
    case parseError
    
    var errorDescription: String? {
        switch self {
        case .invalidFormat:
            return "Invalid or unsupported audio format"
        case .emptyData:
            return "Audio data is empty"
        case .parseError:
            return "Failed to parse audio file"
        }
    }
}

/// Audio preprocessing utility using AVFoundation
/// Automatically handles WAV, SPHERE, M4A, MP3, and other formats via AVAudioFile
public class AudioPreprocessor {
    
    /// Extract normalized PCM samples from audio file data
    /// Uses AVAudioFile which automatically handles all common audio formats
    /// - Parameter audioData: Audio file data
    /// - Returns: Array of normalized float samples (mono, 16kHz, range: -1.0 to 1.0)
    public static func extractPCMSamples(from audioData: Data) throws -> [Float] {
        // Write data to temporary file (AVAudioFile requires a file URL)
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("audio")
        
        try audioData.write(to: tempURL)
        
        defer {
            try? FileManager.default.removeItem(at: tempURL)
        }
        
        // Open audio file with AVAudioFile (handles WAV, SPHERE, M4A, MP3, etc.)
        let audioFile: AVAudioFile
        do {
            audioFile = try AVAudioFile(forReading: tempURL)
        } catch {
            throw AudioPreprocessingError.invalidFormat
        }
        
        // Get file format
        let fileFormat = audioFile.processingFormat
        let frameCount = AVAudioFrameCount(audioFile.length)
        
        guard frameCount > 0 else {
            throw AudioPreprocessingError.emptyData
        }
        
        // Create buffer for reading
        guard let buffer = AVAudioPCMBuffer(pcmFormat: fileFormat, frameCapacity: frameCount) else {
            throw AudioPreprocessingError.parseError
        }
        
        // Read entire file
        try audioFile.read(into: buffer)
        
        // Convert to our target format (16kHz mono Float32)
        let targetSampleRate = 16000.0
        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: targetSampleRate,
            channels: 1,
            interleaved: false
        ) else {
            throw AudioPreprocessingError.parseError
        }
        
        // Convert format if needed
        let convertedBuffer: AVAudioPCMBuffer
        if fileFormat.sampleRate != targetSampleRate || fileFormat.channelCount != 1 {
            guard let converter = AVAudioConverter(from: fileFormat, to: targetFormat) else {
                throw AudioPreprocessingError.invalidFormat
            }
            
            let convertedFrameCount = AVAudioFrameCount(
                Double(frameCount) * (targetSampleRate / fileFormat.sampleRate)
            )
            
            guard let outputBuffer = AVAudioPCMBuffer(
                pcmFormat: targetFormat,
                frameCapacity: convertedFrameCount
            ) else {
                throw AudioPreprocessingError.parseError
            }
            
            var error: NSError?
            let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
                outStatus.pointee = .haveData
                return buffer
            }
            
            converter.convert(to: outputBuffer, error: &error, withInputFrom: inputBlock)
            
            if let error = error {
                throw AudioPreprocessingError.parseError
            }
            
            convertedBuffer = outputBuffer
        } else {
            convertedBuffer = buffer
        }
        
        // Extract float samples
        guard let channelData = convertedBuffer.floatChannelData else {
            throw AudioPreprocessingError.parseError
        }
        
        let sampleCount = Int(convertedBuffer.frameLength)
        let samples = Array(UnsafeBufferPointer(start: channelData[0], count: sampleCount))
        
        return samples
    }
    
    // MARK: - Utility Functions
    
    /// Pad or trim audio samples to a specific length
    public static func padOrTrim(_ samples: [Float], to length: Int) -> [Float] {
        if samples.count > length {
            return Array(samples.prefix(length))
        } else if samples.count < length {
            var padded = samples
            padded.append(contentsOf: [Float](repeating: 0, count: length - samples.count))
            return padded
        }
        return samples
    }
    
    /// Convert float PCM samples to WAV format data
    /// - Parameters:
    ///   - samples: Array of normalized float samples (-1.0 to 1.0)
    ///   - sampleRate: Sample rate in Hz (default: 16000)
    /// - Returns: Complete WAV file data
    public static func convertSamplesToWAV(_ samples: [Float], sampleRate: Int = 16000) -> Data {
        let sampleCount = samples.count
        let byteCount = sampleCount * 2  // 16-bit PCM
        
        var wavData = Data()
        
        // RIFF header
        wavData.append(contentsOf: "RIFF".utf8)
        let fileSize = UInt32(36 + byteCount)
        wavData.append(contentsOf: withUnsafeBytes(of: fileSize.littleEndian) { Data($0) })
        wavData.append(contentsOf: "WAVE".utf8)
        
        // fmt chunk
        wavData.append(contentsOf: "fmt ".utf8)
        let fmtChunkSize = UInt32(16)
        wavData.append(contentsOf: withUnsafeBytes(of: fmtChunkSize.littleEndian) { Data($0) })
        let audioFormat = UInt16(1)  // PCM
        wavData.append(contentsOf: withUnsafeBytes(of: audioFormat.littleEndian) { Data($0) })
        let numChannels = UInt16(1)  // Mono
        wavData.append(contentsOf: withUnsafeBytes(of: numChannels.littleEndian) { Data($0) })
        let sampleRateUInt = UInt32(sampleRate)
        wavData.append(contentsOf: withUnsafeBytes(of: sampleRateUInt.littleEndian) { Data($0) })
        let byteRate = UInt32(sampleRate * 2)  // sampleRate * channels * bytesPerSample
        wavData.append(contentsOf: withUnsafeBytes(of: byteRate.littleEndian) { Data($0) })
        let blockAlign = UInt16(2)  // channels * bytesPerSample
        wavData.append(contentsOf: withUnsafeBytes(of: blockAlign.littleEndian) { Data($0) })
        let bitsPerSample = UInt16(16)
        wavData.append(contentsOf: withUnsafeBytes(of: bitsPerSample.littleEndian) { Data($0) })
        
        // data chunk
        wavData.append(contentsOf: "data".utf8)
        let dataChunkSize = UInt32(byteCount)
        wavData.append(contentsOf: withUnsafeBytes(of: dataChunkSize.littleEndian) { Data($0) })
        
        // Convert float samples to 16-bit PCM
        for sample in samples {
            let clampedSample = max(-1.0, min(1.0, sample))
            let intSample = Int16(clampedSample * 32767.0)
            wavData.append(contentsOf: withUnsafeBytes(of: intSample.littleEndian) { Data($0) })
        }
        
        return wavData
    }
}
