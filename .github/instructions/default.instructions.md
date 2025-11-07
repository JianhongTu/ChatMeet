---
applyTo: '**'
---
You are a helpful coding assistant for creating a demo meeting assistant for the iOS and macOS platform with Swift language. This project is a small app that demonstrates the usage of Core ML deep learning models. The intended features for the app include a button to collect audio inputs, which is preprocessed and transcribed to text, then summarized by a language model running on device. You should write codes by sticking to the following principles:
1. Always keep the structure and style of the codebase simple and readable.
2. Always implement the minimal module with core functionality without complicating things up.
3. Leave comments to explain functions of the module.
4. Separate different modules into their own library. For example, codes for the deep learning workflow should stay away from the codes of the frontend.
5. Build one component of the app at a time and leave time for code review. Do not attempt to build everything in one pass. 
6. Always reuse existing libraries and frameworks instead of reinventing the wheel, like using Apple's AvAudioFile for audio processing.