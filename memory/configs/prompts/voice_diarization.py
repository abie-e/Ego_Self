"""
ASR + Diarization Prompt for Gemini 2.5 Pro
Joint speech recognition + speaker diarization with Gemini.

Adapted from m3-agent.
"""

# System message (Gemini)
SYSTEM_MESSAGE_GEMINI = """You are a professional speech processing expert specializing in Automatic Speech Recognition (ASR) and Speaker Diarization."""

# User prompt for video/audio ASR + diarization (Gemini)
# Reference: m3-agent/mmagent/prompts.py - prompt_audio_segmentation
USER_PROMPT_GEMINI = """You are given a video/audio. Your task is to perform Automatic Speech Recognition (ASR) and audio diarization on the provided media. Extract all speech segments with accurate timestamps and assign speaker identifiers to each segment.

Return a JSON list where each entry represents a speech segment with the following fields:
	•	start_time: Start timestamp in seconds (float).
	•	end_time: End timestamp in seconds (float).
	•	speaker: Speaker identifier in the format "[SPEAKER 1]", "[SPEAKER 2]", etc.
	•	text: The transcribed text for that segment.

Example Output:

[
    {"start_time": 5.0, "end_time": 8.5, "speaker": "[SPEAKER 1]", "text": "Hello, everyone."},
    {"start_time": 9.0, "end_time": 12.3, "speaker": "[SPEAKER 2]", "text": "Welcome to the meeting."}
]

Strict Requirements:

	•	Ensure precise speech segmentation with accurate timestamps in seconds (float).
	•	Segment based on speaker turns and assign speaker identifiers.
	•	Use the format "[SPEAKER 1]", "[SPEAKER 2]", "[SPEAKER 3]", etc. for speaker identifiers.
	•	Preserve punctuation and capitalization in the ASR output.
	•	Skip the speeches that can hardly be clearly recognized.
	•	Return only the valid JSON list (which starts with "[" and ends with "]") without additional explanations.
    •	If the media contains no speech, return an empty list ("[]").
	
Now generate the JSON list based on the given media:"""

