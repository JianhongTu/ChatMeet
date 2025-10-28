//
//  PreviewHelpers.swift
//  MeetingAssistant
//
//  Helper utilities for Xcode previews
//

import SwiftUI

#if DEBUG

/// Mock ViewModel for previews with sample data
@MainActor
class MockMeetingAssistantViewModel: MeetingAssistantViewModel {
    
    static var withSampleData: MockMeetingAssistantViewModel {
        let vm = MockMeetingAssistantViewModel()
        vm.transcription = """
        Welcome to the quarterly planning meeting. Today we'll discuss our goals for Q4 and review the progress from Q3. 
        The main topics include customer satisfaction improvements, new product features, and team expansion plans. 
        John will present the customer feedback analysis, and Sarah will cover the technical roadmap.
        """
        vm.summary = """
        • Quarterly planning meeting for Q4
        • Review Q3 progress and achievements
        • Focus areas: customer satisfaction, product features, team growth
        • Presentations from John (customer feedback) and Sarah (technical roadmap)
        • Action items to be distributed after meeting
        """
        vm.statusMessage = "Complete! Ready to record again"
        return vm
    }
    
    static var recording: MockMeetingAssistantViewModel {
        let vm = MockMeetingAssistantViewModel()
        vm.isRecording = true
        vm.statusMessage = "Recording... Click 'Stop Recording' when finished"
        vm.transcription = ""
        vm.summary = ""
        return vm
    }
    
    static var processing: MockMeetingAssistantViewModel {
        let vm = MockMeetingAssistantViewModel()
        vm.isRecording = false
        vm.statusMessage = "Transcribing audio..."
        vm.transcription = ""
        vm.summary = ""
        return vm
    }
}

#endif
