//
//  AudioPreprocessor.swift
//  ChatMeet
//
//  Basic audio preprocessing utilities for transcription models
//  Extracts and normalizes audio samples from WAV files to mono 16kHz format
//

import Foundation

/// Audio preprocessing errors
enum AudioPreprocessingError: Error, LocalizedError {
    case invalidFormat
    case unsupportedBitDepth
    case unsupportedChannelCount
    case emptyData
    case parseError
    
    var errorDescription: String? {
        switch self {
        case .invalidFormat:
            return "Invalid audio format. Expected WAV file."
        case .unsupportedBitDepth:
            return "Unsupported bit depth. Supported: 16, 24, 32-bit PCM or 32-bit float."
        case .unsupportedChannelCount:
            return "Unsupported channel count."
        case .emptyData:
            return "Audio data is empty."
        case .parseError:
            return "Failed to parse audio file."
        }
    }
}

/// Audio preprocessing utility for extracting and normalizing audio samples
/// Converts WAV files to normalized mono Float32 samples at 16kHz
public class AudioPreprocessor {
    
    /// Extract normalized PCM samples from WAV file data
    /// - Parameter audioData: WAV file data
    /// - Returns: Array of normalized float samples (mono, 16kHz, range: -1.0 to 1.0)
    public static func extractPCMSamples(from audioData: Data) throws -> [Float] {
        // Validate minimum size for WAV header
        guard audioData.count >= 44 else {
            throw AudioPreprocessingError.emptyData
        }
        
        // Parse WAV file structure
        let wavInfo = try parseWAVHeader(audioData)
        
        // Extract raw PCM data
        let pcmData = audioData.subdata(in: wavInfo.dataOffset..<(wavInfo.dataOffset + wavInfo.dataSize))
        
        // Decode PCM based on format
        var samples: [Float]
        
        switch wavInfo.audioFormat {
        case 1:  // PCM format
            samples = try decodePCM(pcmData, 
                                   bitsPerSample: wavInfo.bitsPerSample, 
                                   channels: wavInfo.numChannels)
        case 3:  // IEEE Float format
            samples = try decodeFloat32(pcmData, channels: wavInfo.numChannels)
        default:
            throw AudioPreprocessingError.invalidFormat
        }
        
        // Resample to 16kHz if needed
        if wavInfo.sampleRate != 16000 {
            samples = resampleTo16kHz(samples, fromRate: Float(wavInfo.sampleRate))
        }
        
        return samples
    }
    
    // MARK: - WAV Parsing
    
    private struct WAVInfo {
        let audioFormat: UInt16
        let numChannels: Int
        let sampleRate: UInt32
        let bitsPerSample: Int
        let dataOffset: Int
        let dataSize: Int
    }
    
    private static func parseWAVHeader(_ audioData: Data) throws -> WAVInfo {
        // Verify RIFF header
        let riffMagic = audioData.subdata(in: 0..<4)
        guard String(data: riffMagic, encoding: .ascii) == "RIFF" else {
            throw AudioPreprocessingError.invalidFormat
        }
        
        // Verify WAVE format
        let waveMagic = audioData.subdata(in: 8..<12)
        guard String(data: waveMagic, encoding: .ascii) == "WAVE" else {
            throw AudioPreprocessingError.invalidFormat
        }
        
        // Find fmt and data chunks
        var offset = 12
        var fmtChunkOffset: Int?
        var dataChunkOffset: Int?
        var dataChunkSize: Int?
        
        while offset + 8 <= audioData.count {
            let chunkID = audioData.subdata(in: offset..<(offset + 4))
            let chunkSize = audioData.withUnsafeBytes { bytes in
                bytes.load(fromByteOffset: offset + 4, as: UInt32.self)
            }
            
            let chunkIDString = String(data: chunkID, encoding: .ascii) ?? ""
            
            if chunkIDString == "fmt " {
                fmtChunkOffset = offset + 8
            } else if chunkIDString == "data" {
                dataChunkOffset = offset + 8
                dataChunkSize = Int(chunkSize)
                break
            }
            
            offset += 8 + Int(chunkSize)
        }
        
        guard let fmtOffset = fmtChunkOffset,
              let dataOffset = dataChunkOffset,
              let dataSize = dataChunkSize else {
            throw AudioPreprocessingError.parseError
        }
        
        // Parse fmt chunk
        let audioFormat = audioData.withUnsafeBytes { bytes in
            bytes.load(fromByteOffset: fmtOffset, as: UInt16.self)
        }
        let numChannels = Int(audioData.withUnsafeBytes { bytes in
            bytes.load(fromByteOffset: fmtOffset + 2, as: UInt16.self)
        })
        let sampleRate = audioData.withUnsafeBytes { bytes in
            bytes.load(fromByteOffset: fmtOffset + 4, as: UInt32.self)
        }
        let bitsPerSample = Int(audioData.withUnsafeBytes { bytes in
            bytes.load(fromByteOffset: fmtOffset + 14, as: UInt16.self)
        })
        
        return WAVInfo(
            audioFormat: audioFormat,
            numChannels: numChannels,
            sampleRate: sampleRate,
            bitsPerSample: bitsPerSample,
            dataOffset: dataOffset,
            dataSize: dataSize
        )
    }
    
    // MARK: - PCM Decoding
    
    private static func decodePCM(_ data: Data, bitsPerSample: Int, channels: Int) throws -> [Float] {
        switch bitsPerSample {
        case 16:
            return decodePCM16(data, channels: channels)
        case 24:
            return decodePCM24(data, channels: channels)
        case 32:
            return decodePCM32(data, channels: channels)
        default:
            throw AudioPreprocessingError.unsupportedBitDepth
        }
    }
    
    /// Decode 16-bit PCM and downmix to mono
    private static func decodePCM16(_ data: Data, channels: Int) -> [Float] {
        let sampleCount = data.count / 2 / channels
        var monoSamples = [Float](repeating: 0, count: sampleCount)
        
        data.withUnsafeBytes { (buffer: UnsafeRawBufferPointer) in
            let int16Buffer = buffer.bindMemory(to: Int16.self)
            for i in 0..<sampleCount {
                var sum: Float = 0
                for ch in 0..<channels {
                    sum += Float(int16Buffer[i * channels + ch]) / 32768.0
                }
                monoSamples[i] = sum / Float(channels)
            }
        }
        
        return monoSamples
    }
    
    /// Decode 24-bit PCM and downmix to mono
    private static func decodePCM24(_ data: Data, channels: Int) -> [Float] {
        let sampleCount = data.count / 3 / channels
        var monoSamples = [Float](repeating: 0, count: sampleCount)
        
        data.withUnsafeBytes { (buffer: UnsafeRawBufferPointer) in
            for i in 0..<sampleCount {
                var sum: Float = 0
                for ch in 0..<channels {
                    let offset = (i * channels + ch) * 3
                    let byte1 = Int32(buffer[offset])
                    let byte2 = Int32(buffer[offset + 1])
                    let byte3 = Int32(buffer[offset + 2])
                    var sample24 = (byte3 << 16) | (byte2 << 8) | byte1
                    if sample24 & 0x800000 != 0 {
                        sample24 |= Int32(bitPattern: 0xFF000000)
                    }
                    sum += Float(sample24) / 8388608.0
                }
                monoSamples[i] = sum / Float(channels)
            }
        }
        
        return monoSamples
    }
    
    /// Decode 32-bit PCM and downmix to mono
    private static func decodePCM32(_ data: Data, channels: Int) -> [Float] {
        let sampleCount = data.count / 4 / channels
        var monoSamples = [Float](repeating: 0, count: sampleCount)
        
        data.withUnsafeBytes { (buffer: UnsafeRawBufferPointer) in
            let int32Buffer = buffer.bindMemory(to: Int32.self)
            for i in 0..<sampleCount {
                var sum: Float = 0
                for ch in 0..<channels {
                    sum += Float(int32Buffer[i * channels + ch]) / 2147483648.0
                }
                monoSamples[i] = sum / Float(channels)
            }
        }
        
        return monoSamples
    }
    
    /// Decode 32-bit float PCM and downmix to mono
    private static func decodeFloat32(_ data: Data, channels: Int) throws -> [Float] {
        let sampleCount = data.count / 4 / channels
        var monoSamples = [Float](repeating: 0, count: sampleCount)
        
        data.withUnsafeBytes { (buffer: UnsafeRawBufferPointer) in
            let floatBuffer = buffer.bindMemory(to: Float.self)
            for i in 0..<sampleCount {
                var sum: Float = 0
                for ch in 0..<channels {
                    sum += floatBuffer[i * channels + ch]
                }
                monoSamples[i] = sum / Float(channels)
            }
        }
        
        return monoSamples
    }
    
    // MARK: - Resampling
    
    /// Simple linear resampling to 16kHz
    private static func resampleTo16kHz(_ samples: [Float], fromRate: Float) -> [Float] {
        if fromRate == 16000.0 {
            return samples
        }
        
        let ratio = fromRate / 16000.0
        let outputLength = Int(Float(samples.count) / ratio)
        var resampled = [Float](repeating: 0, count: outputLength)
        
        for i in 0..<outputLength {
            let srcPos = Float(i) * ratio
            let srcIdx = Int(srcPos)
            let frac = srcPos - Float(srcIdx)
            
            if srcIdx + 1 < samples.count {
                resampled[i] = samples[srcIdx] * (1.0 - frac) + samples[srcIdx + 1] * frac
            } else if srcIdx < samples.count {
                resampled[i] = samples[srcIdx]
            }
        }
        
        return resampled
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
}
