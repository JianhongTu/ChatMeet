//
//  PerformanceStatsView.swift
//  ChatMeet
//
//  Detailed performance statistics view
//

import SwiftUI

struct PerformanceStatsView: View {
    let recordingDuration: TimeInterval
    let transcriptionTime: TimeInterval
    let timeToFirstToken: TimeInterval
    let summarizationTime: TimeInterval
    let tokensPerSecond: Double
    let realTimeFactor: Double
    let totalProcessingTime: TimeInterval
    
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Performance Statistics")
                    .font(.title2)
                    .fontWeight(.semibold)
                Spacer()
                Button("Close") {
                    dismiss()
                }
            }
            .padding()
            .background(Color.secondary.opacity(0.1))
            
            ScrollView {
                VStack(spacing: 20) {
                    // Main metrics
                    VStack(spacing: 16) {
                        metricRow(
                            icon: "mic.fill",
                            label: "Recording Duration",
                            value: formatTime(recordingDuration),
                            color: .blue
                        )
                        
                        Divider()
                        
                        metricRow(
                            icon: "waveform",
                            label: "Transcription Time",
                            value: formatTime(transcriptionTime),
                            color: .green
                        )
                        
                        Divider()
                        
                        metricRow(
                            icon: "timer",
                            label: "Time to First Token",
                            value: formatTime(timeToFirstToken),
                            color: .orange,
                            subtitle: "Latency before first output"
                        )
                        
                        Divider()
                        
                        metricRow(
                            icon: "text.bubble.fill",
                            label: "Summarization Time",
                            value: formatTime(summarizationTime),
                            color: .purple
                        )
                    }
                    .padding()
                    .background(Color.secondary.opacity(0.05))
                    .cornerRadius(12)
                    
                    // Performance metrics
                    VStack(spacing: 16) {
                        metricRow(
                            icon: "gauge.with.dots.needle.67percent",
                            label: "Real-Time Factor",
                            value: String(format: "%.2fx", realTimeFactor),
                            color: realTimeFactor < 1.0 ? .green : (realTimeFactor < 2.0 ? .orange : .red),
                            subtitle: realTimeFactor < 1.0 ? "Faster than real-time" : "Slower than real-time"
                        )
                        
                        Divider()
                        
                        metricRow(
                            icon: "speedometer",
                            label: "Tokens per Second",
                            value: String(format: "%.1f tok/s", tokensPerSecond),
                            color: .cyan,
                            subtitle: "Summarization throughput"
                        )
                        
                        Divider()
                        
                        metricRow(
                            icon: "clock.fill",
                            label: "Total Processing Time",
                            value: formatTime(totalProcessingTime),
                            color: .blue,
                            subtitle: "End-to-end duration",
                            highlighted: true
                        )
                    }
                    .padding()
                    .background(Color.secondary.opacity(0.05))
                    .cornerRadius(12)
                }
                .padding()
            }
        }
        .frame(width: 500, height: 550)
    }
    
    private func metricRow(
        icon: String,
        label: String,
        value: String,
        color: Color,
        subtitle: String? = nil,
        highlighted: Bool = false
    ) -> some View {
        HStack(spacing: 16) {
            Image(systemName: icon)
                .font(.system(size: 24))
                .foregroundColor(color)
                .frame(width: 40)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(label)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                if let subtitle = subtitle {
                    Text(subtitle)
                        .font(.caption)
                        .foregroundColor(.secondary.opacity(0.8))
                }
            }
            
            Spacer()
            
            Text(value)
                .font(.system(size: highlighted ? 20 : 18, weight: highlighted ? .bold : .semibold))
                .foregroundColor(highlighted ? color : .primary)
        }
    }
    
    private func formatTime(_ time: TimeInterval) -> String {
        if time < 1.0 {
            return String(format: "%.0f ms", time * 1000)
        } else if time < 60.0 {
            return String(format: "%.2f s", time)
        } else {
            let minutes = Int(time) / 60
            let seconds = Int(time) % 60
            return String(format: "%d:%02d", minutes, seconds)
        }
    }
}

#Preview {
    PerformanceStatsView(
        recordingDuration: 30.5,
        transcriptionTime: 5.2,
        timeToFirstToken: 0.8,
        summarizationTime: 2.1,
        tokensPerSecond: 15.3,
        realTimeFactor: 0.17,
        totalProcessingTime: 38.6
    )
}
