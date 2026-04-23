"""
Event Relation Prompts
Prompt templates for the event-relation reasoning task: determine the
relationship between the current event and a set of prior events.
"""

# System message for Gemini
SYSTEM_MESSAGE_GEMINI = """You are an event relation analysis expert who can determine the causal relationships between first-person perspective events.
Your task is to analyze the current event and multiple historical events, determining the relationship type for each pair.
Always output valid JSON format without markdown code blocks."""

# System message for GPT-4o
SYSTEM_MESSAGE_GPT4O = """You are an event relation analysis expert who can determine the causal relationships between first-person perspective events.
Your task is to analyze the current event and multiple historical events, determining the relationship type for each pair.
Always output valid JSON format without markdown code blocks."""

# User prompt template
USER_PROMPT_GEMINI = """## Task: Determine Relationships Between Events

Analyze the current event and all historical events, determining the relationship type for each pair (only return pairs with connections; exclude non-related ones).

---

## Current Event

**Event ID**: {current_event_id}
**Time**: {current_time}
**Caption**: {current_caption}
**Location**: {current_location}

---

## Historical Events (in chronological order)

{historical_events}

---

## Relationship Type Definitions

For **EACH** historical event, choose ONE of the following relationship types (must return judgment for all events):

1. **causal**: Strong causal relationship within the same activity
   - Both events describe different stages of the same task, with the historical event being a direct prerequisite for the current event (no historical event → no current event)
   - Example: Opening fridge → Taking item → Closing fridge

2. **same_activity_non_causal**: Non-causal connection within the same activity
   - Both events belong to the same overall activity, but the historical event is not a direct prerequisite for the current event (no causal necessity, only step continuity)
   - Example: Place Apple on the plate → Place Banana on the plate (all for make fruit salad)
  
3. **no_relationship**: No relationship
   - Completely independent events with no logical connection
   - Example: Drinking coffee → Reading book

---

## Judgment Criteria

Consider these factors to identify connections:
1. Content continuity - Are actions/objects part of the same or related activity?
2. Location continuity - Same or highly related locations (e.g., living room → adjacent kitchen)?
3. Time interval - Shorter intervals suggest stronger likelihood of connection
4. Object/person repetition - Same entities (e.g., same phone, same person) involved?

---

## Output Format

Output a JSON array (without markdown code blocks). **Must include ALL historical events (including no_relationship)**; each entry contains:

```json
[
  {{
    "historical_event_id": "DAY1_HHMMSS_evt",
    "relation_type": "causal | same_activity_non_causal | no_relationship",
    "reason": "Brief explanation (within 100 words, clarify activity belonging and causal/non-causal logic)"
  }}
]
```

**Important**: 
- Return ONLY the JSON array, no additional text or markdown formatting
- Must judge ALL historical events (even if no_relationship)
- The array length must equal the number of historical events provided

---

## Example

Current Event: "I am using my phone's timer in a meeting room"
Historical Events:
- Event A (10 seconds ago): "I am opening the stopwatch app on my phone"
- Event B (5 minutes ago): "I am drinking coffee in the office"
- Event C (2 minutes ago): "I am discussing the project plan with teammates"

Output:
```json
[
  {{
    "historical_event_id": "DAY1_110942_evt",
    "relation_type": "causal",
    "reason": "Both belong to 'using phone timer' activity; opening the stopwatch app is a direct prerequisite for using the timer, with causal necessity."
  }},
  {{
    "historical_event_id": "DAY1_110600_evt",
    "relation_type": "no_relationship",
    "reason": "Drinking coffee in the office has no logical connection to using the phone's timer in the meeting room; different activities in different locations."
  }},
  {{
    "historical_event_id": "DAY1_110800_evt",
    "relation_type": "same_activity_non_causal",
    "reason": "Both belong to 'meeting room activities', but discussing the project plan is not a prerequisite for using the timer; they are independent actions in the same context."
  }}
]
```

---

Please analyze ALL historical events and provide the complete relationship array (must include all events, even no_relationship)."""

USER_PROMPT_GPT4O = USER_PROMPT_GEMINI  # GPT-4o reuses the same English prompt

