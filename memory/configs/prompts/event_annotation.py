"""
Event Annotation Prompts
Prompt templates for the event annotation task.
"""

# System messages: define the assistant's role and behavior
SYSTEM_MESSAGE_GPT4O = """You are an expert at analyzing first-person egocentric videos. 
Your task is to generate detailed, structured event descriptions focusing on what the wearer ("I") is doing and their interactions with people and objects.
Always output valid JSON format without markdown code blocks."""

SYSTEM_MESSAGE_GEMINI = """You are an expert at analyzing first-person egocentric videos and associated audio. 
Your task is to generate detailed, structured event descriptions based on both visual and auditory information, focusing on what the wearer ("I") is doing.
Always output a single, valid JSON object without markdown code blocks."""

SYSTEM_MESSAGE_GPT5 = """You are an expert at analyzing first-person egocentric videos and associated audio. 
Your task is to generate detailed, structured event descriptions based on both visual and auditory information, focusing on what the wearer ("I") is doing.
Always output a single, valid JSON object without markdown code blocks."""

# User prompts: task instructions and output format
USER_PROMPT_GPT4O = """## Task: Egocentric Video Event Extraction

Analyze the first-person video frames to generate a detailed, structured summary of the wearer's ("I") actions. Focus on meaningful interactions related to the main event, filtering out trivialities.

---

## Reasoning Steps (Chain-of-Thought)

Carefully consider and answer these steps in order:

1. **Environment**: Where am I? What kind of place or environment? Identify the scene type (e.g., kitchen, office).
2. **Interaction Targets**: Who and what am I meaningfully interacting with? Identify the key people and up to two most important objects involved in the main event. Note their appearance, features, and location.
3. **Sequence of Events**: What are the interactions happening over time? For each target, list the actions chronologically.
4. **Caption Summary**: Synthesize the key events into a single, comprehensive first-person sentence capturing all interactions (both my actions and actions directed toward me). Use <humanX> names for people and detailed category descriptions for objects.

---

## Output Format

Output ONLY a single valid JSON object (no markdown ```json``` wrapper):

```json
{
  "caption": "A first-person episodic memory description capturing all interactions (both my actions and actions directed toward me). For people: use <humanX> names. For objects: use detailed category descriptions.",
  "interaction_target": [
    {
      "name": "string, a unique identifier, must be 'humanX' or 'objectX' (e.g., 'human1', 'object1')",
      "action": "string, verb phrase in first-person present tense (max 10 words), no subject, describing the interaction",
      "description": "string, templated description following specific format (see below)",
      "location": "string, spatial location of this target relative to me or the environment",
      "interaction_segments": [
        {
          "start_time": "float, start timestamp (in seconds) of an interaction period with this target",
          "end_time": "float, end timestamp (in seconds) of an interaction period with this target"
        }
      ]
    }
  ],
  "interaction_location": "string, the type of scene or location with 1-2 key descriptive features.",
  "interaction_language": [
    {
      "start_time": "float, original start time from input, unchanged",
      "end_time": "float, original end time from input, unchanged",
      "speaker": null,
      "text": "string, original transcription text from input, unchanged",
      "description": null,
      "reason": null
    }
  ]
}
```

---

## Field Requirements

### caption
- A single sentence describing the core event from my first-person perspective, capturing all interactions happening in this scene (both my actions toward others/objects and actions directed toward me).
- Structure: **where I am + what I'm doing/saying + what others are doing to me** (if applicable).
- Must be in the first person, starting with "I am..." or "I'm...".
- For people: use interaction_target names wrapped in angle brackets (e.g., `<human1>`, `<human2>`).
- For objects: use the object's detailed category description (not the `<objectX>` name).
- Examples: 
  - "I am in the office discussing the project timeline with <human1> while reviewing documents on a white rectangular laptop."

### interaction_target
- **Important**: Only include truly significant interaction targets that meet ALL of these criteria:
  - **Interaction duration**: I have direct interaction (language or behavior) with them for over 5 seconds.
  - **Event impact**: If removed from the description, the core event would be incomplete or meaningfully changed.
  - **Individual entity**: Each entry must be a single, unique person or object, never a group or aggregate.
  - **Max count**: 3 humans, 2 key objects (central to the main event).
- Exclude background people/objects and those only briefly touched or not directly engaged.
- The list must be sorted by the time of the first interaction.
- Each item has four fields: `name`, `action`, `description`, `location`, and `interaction_segments`.
  - `name`: Must follow the format `humanX` or `objectX` (e.g., "human1", "object1", "object2").
  - `action`: A verb phrase in first-person present tense (max 10 words), no subject. Describes my interaction with this target.
    - For conversation: use phrases like "discuss breakfast plans with", "talk about the meeting with"
    - For physical actions: must include the object's category as the object of the verb (e.g., "pour orange juice from glass pitcher", "pick up and examine coffee mug")
    - Examples: "pour orange juice into glass cup", "discuss work schedule with", "pick up and open laptop"
  - `description`: A templated description following strict format:
    - **For objects**: `[color] [shape] [size] [material] [precise object category] with [distinctive long-term visual features]`
      - All components are required. The "distinctive long-term visual features" must be specific and detailed.
      - Example: "black rectangular medium-sized plastic coffee machine with a silver metal carafe and a round red power button on the front panel"
      - Example: "transparent cylindrical medium-sized glass pitcher with orange juice inside, a curved plastic handle, and a silver metal lid"
      - Be as precise as possible with the object category (e.g., "French press coffee maker" instead of just "coffee maker")
    - **For people**: `[hair color] [hairstyle] [gender] person with [clothing colors and styles from top to bottom], [accessories with colors and styles if any], [age description: young/middle-aged/elderly], [body type: height and build], with [distinctive long-term visual features]`
      - All components are required. If no accessories, write "no accessories". The "distinctive long-term visual features" must be specific.
      - Example: "brown short-haired male person with blue collared shirt and black jeans, silver-framed glasses, middle-aged, tall and slim build, with a small scar above left eyebrow"
      - Example: "black long-haired female person with red knit sweater and blue jeans, no accessories, middle-aged, average height and slim build, with a gentle smile and round face"
    - The description must enable unique identification of this object or person in the video.
    - **Important**: The description should only include appearance details and must not contain any action or interaction-related information.
  - `location`: Describe the spatial location of this target relative to me or the environment.
    - Examples: "on the white dining table in front of me", "standing to my right near the window", "in my left hand", "on the kitchen counter to my left"
  - `interaction_segments`: A list of time segments (each with `start_time` and `end_time` in seconds as floats) during which I am actively interacting with this target. Each segment represents a continuous period of meaningful interaction. Segments should be sorted chronologically. Only include periods where the target is clearly visible and directly engaged with me (not just present in the background).

  **Important for people**: Only individuals who have important, direct, and sustained (over 5 seconds, one-on-one) interaction with me are included. All others must be omitted.

#### Negative Example

Below is a counterexample from the raw data. It demonstrates common mistakes regarding the `interaction_target` construction:

```json
{
  "name": "human1",
  "description": "A group of five women sitting around a table with a red and white checkered tablecloth.",
  "action": "observe me as I prepare to start a timer",
  "location": "around the table"
}
```

**Analysis of This Negative Example:**
- The "human1" here refers to an entire group, not a single, uniquely identifiable individual. This violates the rule that each human entry must reference only a single instance, not a group or aggregate.
- The interaction described ("observing me as I prepare to start a timer") is not a direct, one-on-one, sustained engagement with any specific individual. Instead, it describes background or passive observation by a group, which must not be included in the `interaction_target`.
- As per the core requirements, only individual persons with important, direct, and sustained (over 5 seconds, one-on-one) interactions should be included, and all groups or indirect participants should be omitted.

---

### interaction_location
- The type of location or scene, with 1-3 key descriptive features of the environment.
- The description should be concise.
- Examples: `meeting room with a large whiteboard`, `kitchen with stainless steel appliances`.

### interaction_language

You will be provided with an input JSON object containing `speech_segments`. Simply copy this data to the output without modification, but set `speaker` and `description` to `null` for all segments.

**Input Data:**
{speech_segments_json}

**Output Requirements:**
- This is a pass-through of the input `speech_segments` with `speaker`, `description`, and `reason` set to `null`.
- It must be a list of segment objects.
- **DO NOT** modify `start_time`, `end_time`, or `text`.
- Set `speaker`: `null` for all segments.
- Set `description`: `null` for all segments.
- Set `reason`: `null` for all segments.

---

## Example

```json
{
  "caption": "I am in the kitchen pouring orange juice from a transparent cylindrical glass pitcher into a tall clear glass cup while <human1> sits across from me.",
  "interaction_target": [
    {
      "name": "object1",
      "action": "pour orange juice from glass pitcher",
      "description": "transparent cylindrical medium-sized glass pitcher with orange juice inside, a curved plastic handle, and a silver metal lid",
      "location": "on the white dining table in front of me",
      "interaction_segments": [
        {
          "start_time": 9.3,
          "end_time": 12.5
        }
      ]
    },
    {
      "name": "object2",
      "action": "pour orange juice into glass cup",
      "description": "transparent cylindrical tall glass cup with smooth surface and no patterns",
      "location": "in my left hand",
      "interaction_segments": [
        {
          "start_time": 10.1,
          "end_time": 12.5
        }
      ]
    },
    {
      "name": "human1",
      "action": "talk about breakfast plans with",
      "description": "black long-haired female person with red knit sweater and blue jeans, no accessories, middle-aged, average height and slim build, with a gentle smile and round face",
      "location": "sitting across the dining table from me",
      "interaction_segments": [
        {
          "start_time": 8.7,
          "end_time": 15.0
        }
      ]
    }
  ],
  "interaction_location": "kitchen with white cabinets and stainless steel appliances",
  "interaction_language": [
    {
      "start_time": 10.5,
      "end_time": 12.8,
      "speaker": null,
      "text": "let's have breakfast together",
      "description": null,
      "reason": null
    },
    {
      "start_time": 13.2,
      "end_time": 14.1,
      "speaker": null,
      "text": "sounds good",
      "description": null,
      "reason": null
    }
  ]
}
```

---

## Critical Rules

1. Output ONLY valid JSON (no ```json``` markdown).
2. The `interaction_target` list must be sorted by the first interaction time.
3. The `action` field within each `interaction_target` item must be a verb phrase in first-person present tense (max 10 words), no subject.
4. The `description` field must strictly follow the templated format for objects or people.
5. The `location` field must specify the spatial location of each target.
6. For `interaction_language`: copy input data exactly, only set `speaker`, `description`, and `reason` to `null`.
7. The `caption` must capture all interactions (both my actions and actions directed toward me). Use <humanX> names for people and detailed category descriptions for objects.
8. Focus on meaningful interactions. The `interaction_target` list should include a maximum of two objects.
9. For every `interaction_target`, you MUST include the `interaction_segments` field as a list of time periods. Each segment has `start_time` and `end_time` (floats in seconds). Only include periods where I am actively interacting with the target (not just when it's visible in the background).

Now analyze the input video and speech segments to generate the JSON output."""

USER_PROMPT_GEMINI = """## Task: Egocentric Video Event Extraction

Analyze the first-person video frames and audio transcription to generate a detailed, structured summary of the wearer's ("I") actions. Your analysis must be based on a holistic understanding of both visual and auditory information. Only focus on meaningful interactions related to the main event, filtering out trivialities. You will also correct the transcription and perform speaker diarization.

---

## Reasoning Steps (Chain-of-Thought)

Carefully consider and answer these steps in order:

1.  **Environment**: Where am I? What kind of place or environment? Identify the scene type (e.g., kitchen, office).
2.  **Interaction Targets**: Who and what am I meaningfully interacting with? Identify the key people and up to two most important objects involved in the main event. Note their appearance, features, and location.
3.  **Sequence of Events**: What are the interactions happening over time? For each target, list the actions and conversations chronologically.
4.  **Caption Summary**: Synthesize the key events into a single, comprehensive first-person sentence following the format: where I am, what kind of objects/people I'm with, and what I'm doing/saying.
5.  **Transcription Correction & Diarization**: Review the provided `speech_segments`. Correct any transcription errors based on the video's audio. Identify the speaker for each segment using the names of the interaction targets.

---

## Output Format

Output ONLY a single valid JSON object (no markdown ```json``` wrapper):

```json
{
  "caption": "A first-person episodic memory description capturing all interactions (both my actions and actions directed toward me). For people: use <humanX> names. For objects: use detailed category descriptions.",
  "interaction_target": [
    {
      "name": "string, a unique identifier, must be 'humanX' or 'objectX' (e.g., 'human1', 'object1')",
      "action": "string, a single sentence (up to 15 words) that clearly describes, in detail, the way I interact with this target, focusing on the interaction itself and including necessary details for clarity.",
      "description": "string, templated description following specific format (see below)",
      "location": "string, spatial location of this target relative to me or the environment",
      "interaction_segments": [
        {
          "start_time": "float, start timestamp (in seconds) of an interaction period with this target",
          "end_time": "float, end timestamp (in seconds) of an interaction period with this target"
        }
      ]
    }
  ],
  "interaction_location": "string, the type of scene or location with 1-2 key descriptive features.",
  "interaction_language": [
    {
      "start_time": "float, original start time, unchanged",
      "end_time": "float, original end time, unchanged",
      "speaker": "string, name of the speaker from interaction_target (e.g., 'human1', 'I')",
      "text": "string, corrected transcription of the speech (only minor corrections allowed).",
      "description": "string, the detailed description of the speaker, copied from the corresponding 'interaction_target' entry.",
      "reason": "string, one-sentence explanation of why this person is identified as the speaker, based on gender, actions, mouth movements, and context."
    }
  ]
}
```

---
 
## Field Requirements

### caption
- A single sentence describing the core event from my first-person perspective, capturing all interactions happening in this scene (both my actions toward others/objects and actions directed toward me).
- Structure: **where I am + what I'm doing/saying + what others are doing to me** (if applicable).
- Must be in the first person, starting with "I am..." or "I'm...".
- For people: use interaction_target names wrapped in angle brackets (e.g., `<human1>`, `<human2>`).
- For objects: use the object's detailed category description (not the `<objectX>` name).
- Should include key spoken dialogue if present.
- Examples: 
  - "I am in the office discussing the project timeline with <human1> who says to me 'The meeting is at 2 PM' while reviewing documents on a white rectangular laptop."

### interaction_target
- **Important**: Only include truly significant interaction targets that meet ALL of these criteria:
  - **Interaction duration**: I have direct interaction (language or behavior) with them for over 5 seconds.
  - **Event impact**: If removed from the description, the core event would be incomplete or meaningfully changed.
  - **Individual entity**: Each entry must be a single, unique person or object, never a group or aggregate.
- Exclude background people/objects and those only briefly touched or not directly engaged.
- The list must be sorted by the time of the first interaction.
- Each item has five fields: `name`, `action`, `description`, `location`, and `interaction_segments`.
  - `name`: Must follow the format `humanX` or `objectX` (e.g., "human1", "object1", "object2").
    - Max: 3 humans, 2 key objects (central to the main event).
  - `action`: Write a full sentence (no more than 15 words) that clearly and specifically explains how I interact with this person or object, focusing on the details and context of the interaction itself. **Do not use just a verb phrase; your sentence must explain what I do with/to this target, in first person present tense, and be as explicit as possible about the interaction.**
      - For conversation: Use a full sentence stating the main conversational action/nature, e.g., "I discuss breakfast plans with her and respond to her invitation with a smile."
      - For physical objects: Use a full sentence describing exactly what I do with the object, e.g., "I pour orange juice from the glass pitcher into the tall clear glass while standing at the table."
  - `description`: A concise appearance-focused description (15 words or less) that always begins with the object or person itself:
    - **For objects**: The description **must include all** of the following components (no omissions allowed): `[color] [shape] [size] [precise object category] with [distinctive long-term visual features]`
      - *All fields are required. Missing any of these components is not allowed.*
      - Example: "black rectangular medium-sized coffee machine with a silver metal carafe and a round red power button on the front panel"
    - **For people**: The description **must include all** of the following components (no omissions allowed): `[hair color] [hairstyle] [gender] person with [clothing colors and styles from top to bottom], [accessories with colors and styles if any], with [distinctive long-term visual features]`
      - *All fields are required. Missing any of these components is not allowed. Please provide a detailed description of clothing (colors and styles, top to bottom)*
      - Example: "brown short-haired male person with blue collared shirt and black jeans, silver-framed glasses, tall and slim build, with a small scar above left eyebrow"
    - **Note:** All descriptions must maximize detail about objects and people, but never exceed 15 words or include any actions or interaction info.
  - `location`: Describe the spatial location of this target relative to me or the environment.
    - Examples: "on the white dining table in front of me", "standing to my right near the window", "in my left hand", "on the kitchen counter to my left"
  - `interaction_segments`: A list of time segments (each with `start_time` and `end_time` in seconds as floats) during which I am actively interacting with this target. Each segment represents a continuous period of meaningful interaction. Segments should be sorted chronologically. Only include periods where the target is clearly visible and directly engaged with me (not just present in the background).

  **Important for people**: Only individuals who have important, direct, and sustained (over 5 seconds, one-on-one) interaction with me are included. All others must be omitted.

---

### interaction_location
- The type of location or scene, with 1-3 key descriptive features of the environment.
- The description should be concise.
- Examples: `meeting room with a large whiteboard`, `kitchen with stainless steel appliances`.

### interaction_language

You will be provided with an input JSON object containing `speech_segments`. Analyze both the video (especially mouth movements) and audio to correct and diarize this data.

**Input Data:**
{speech_segments_json}

**Output Requirements:**
- This is the corrected and diarized version of the input `speech_segments`.
- The output **must be a list of segment objects**.
- For each segment:
  - `text`: Perform only **minor corrections** to the original transcription based on the audio. Do not rewrite sentences.
  - `speaker`: Carefully observe the video, focus on people's movements and mouth movements, and identify the speaker for each segment. Use the name from interaction_target (e.g., "human1"). If I (the camera wearer) am speaking, use "I".
  - `description`: Add the description of the speaker, copied from the corresponding `interaction_target` entry. This should be consistent for the same person.
  - `reason`: Provide a one-sentence explanation of why this person is identified as the speaker. Consider: (1) gender match with voice tone, (2) mouth movements visible in video, (3) body language and gestures, (4) contextual clues from conversation flow. Example: "Female voice matches human1's gender, and her mouth movements are clearly visible during this speech."
- **Important**: Do NOT add any segments and Do Not Change the original timestamps. 

---

## Example

{
  "caption": "I am in the kitchen pouring orange juice from a transparent cylindrical glass pitcher into a tall clear glass cup while <human1> sits across from me and says 'let's have breakfast together', and I reply 'sounds good'.",
  "interaction_target": [
    {
      "name": "object1",
      "action": "I pour orange juice from the transparent glass pitcher into a tall clear cup in front of me.",
      "description": "transparent cylindrical medium-sized glass pitcher with orange juice inside, a curved plastic handle, and a silver metal lid",
      "location": "on the white dining table in front of me",
      "interaction_segments": [
        { "start_time": 9.3, "end_time": 12.5 }
      ]
    },
    {
      "name": "human1",
      "action": "I discuss breakfast options with her and agree to eat together after her suggestion.",
      "description": "black long-haired female person with red knit sweater and blue jeans, with a gentle smile and round face",
      "location": "sitting across the dining table from me",
      "interaction_segments": [
        { "start_time": 8.7, "end_time": 15.0 }
      ]
    }
  ],
  "interaction_location": "kitchen with white cabinets and stainless steel appliances",
  "interaction_language": [
    {
      "start_time": 10.5,
      "end_time": 12.8,
      "speaker": "human1",
      "text": "let's have breakfast together",
      "description": "black long-haired female person with red knit sweater and blue jeans, with a gentle smile and round face",
      "reason": "Female voice matches human1's gender, and her mouth movements are clearly visible during this speech."
    },
    {
      "start_time": 13.2,
      "end_time": 14.1,
      "speaker": "I",
      "text": "sounds good",
      "description": "I",
      "reason": "Voice from camera wearer's perspective."
    }
  ]
}
}
```

---

## Critical Rules

1.  Output ONLY valid JSON (no ```json``` markdown).
2.  The `interaction_target` list must be sorted by the first interaction time.
3.  The `description` field must strictly follow the templated format for objects or people.
4.  The `location` field must specify the spatial location of each target.
5.  Speaker names in `interaction_language` must match the `name` field in `interaction_target`. Use "I" for the camera wearer.
6.  The `caption` must follow the format: where I am + what kind of objects/people + what I'm doing/saying.
7.  Focus on meaningful interactions. The `interaction_target` list should include a maximum of two objects.
8.  For every `interaction_target`, you MUST include the `interaction_segments` field as a list of time periods. Each segment has `start_time` and `end_time` (floats in seconds). Only include periods where I am actively interacting with the target (not just when it's visible in the background).

Now analyze the input video and speech segments to generate the JSON output.
"""

# GPT-5 reuses the Gemini prompt (both support direct video + audio input)
USER_PROMPT_GPT5 = USER_PROMPT_GEMINI

