#!/usr/bin/env python3
"""
Single-video processing script with multi-step pipeline support.

Supported steps:
1. asr: speech recognition (extract audio from the video and run ASR)
2. caption: event annotation (generate the event description)
3. voiceprint: speaker identification (match speakers against a global DB)
4. relation: event-relation reasoning (temporal / causal links to prior events)
5. entity: object tracking + feature extraction (detect, track interaction targets,
           extract features, and globally match across events)

Usage:
python scripts/run_single_video.py \
  --video /path/to/video.mp4 \
  --config configs/config.yaml
"""

import argparse
import sys
import os
import json
from pathlib import Path

# Add src/ to PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import Config
from event import EventAnnotator
from event.event_relation_processor import EventRelationProcessor
from voice.asr import create_asr_processor
from voice.voiceprint import VoiceprintProcessor
from utils.audio_utils import extract_audio_from_video
from utils import parse_video_filename, generate_event_id
from entity.entity_tracker_main import process_event_tracking, run_entity_step
from entity.global_object_matcher import GlobalObjectMatcher


def run_asr_step(config: Config, video_path: str, event_id: str) -> str:
    """
    Step 1: ASR (speech recognition)

    Pipeline:
    1. Extract audio from the video to a temp file
    2. Run the ASR processor
    3. Save the result to output_root/voices/asr/{event_id}.json

    Args:
        config: Config object
        video_path: input video path
        event_id: event ID

    Returns:
        Path to the ASR result JSON.
    """
    print(f"\n{'='*70}")
    print(" Step 1: ASR (speech recognition)")
    print(f"{'='*70}")

    # 1. Prepare output path
    asr_dir = Path(config.voiceprint_asr_dir)
    asr_dir.mkdir(parents=True, exist_ok=True)
    asr_output_path = asr_dir / f"{event_id}.json"

    # Skip if already done
    if asr_output_path.exists():
        print(f"✓ ASR result already exists, skipping")
        return str(asr_output_path)

    # 2. Extract audio to a temp directory
    temp_dir = Path(config.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_audio_path = temp_dir / f"{event_id}.wav"

    print(f"Extracting audio...")
    extract_audio_from_video(video_path, str(temp_audio_path))

    # 3. Build ASR processor
    use_gemini = (config.asr_client.lower() == "gemini")
    asr_processor = create_asr_processor(config, use_gemini=use_gemini)

    # 4. Run ASR
    print(f"Running ASR...")
    asr_result = asr_processor.process(
        audio_path=str(temp_audio_path),
        event_id=event_id
    )

    # 5. Save result
    with open(asr_output_path, 'w', encoding='utf-8') as f:
        json.dump(asr_result, f, ensure_ascii=False, indent=2)

    print(f"✓ ASR done, found {len(asr_result['speech_segments'])} speech segments")

    # Note: keep the temp audio for the voiceprint step; cleaned up at the end of the run.

    return str(asr_output_path)


def run_caption_step(config: Config, video_path: str, asr_output_path: str, event_id: str) -> str:
    """
    Step 2: Caption (event annotation)

    Pipeline:
    1. Load the ASR result
    2. Run EventAnnotator to generate an event description
    3. Save the event JSON to output_root/events/{day}/{event_id}.json

    Args:
        config: Config object
        video_path: input video path
        asr_output_path: ASR result path
        event_id: event ID

    Returns:
        Path to the event JSON.
    """
    print(f"\n{'='*70}")
    print(" Step 2: Caption (event annotation)")
    print(f"{'='*70}")

    # 1. Load ASR result
    speech_segments = None
    if asr_output_path and os.path.exists(asr_output_path):
        with open(asr_output_path, 'r', encoding='utf-8') as f:
            asr_result = json.load(f)
            speech_segments = asr_result.get("speech_segments")
        print(f"Loaded {len(speech_segments) if speech_segments else 0} speech segments")

    # 2. Build EventAnnotator
    annotator = EventAnnotator(config)

    # 3. Annotate
    print(f"Generating event annotation...")
    event_ids = annotator.annotate_video(video_path, speech_segments_json=speech_segments)

    # 4. Resolve event JSON path
    actual_event_id = event_ids[0] if event_ids else event_id
    video_info = parse_video_filename(video_path)
    day = video_info['day']
    event_path = Path(config.events_dir) / f"DAY{day}" / f"{actual_event_id}.json"

    print(f"✓ Caption done")

    return str(event_path)


def run_voiceprint_step(config: Config, asr_output_path: str, event_path: str, event_id: str, video_path: str = None):
    """
    Step 3: Voiceprint (speaker identification)

    Pipeline:
    1. Load the ASR result
    2. Ensure the audio file exists (extract from video if needed)
    3. Run VoiceprintProcessor to match speakers
    4. Update speaker fields in the ASR file
    5. Update interaction_language speaker fields in the event JSON
    6. Update the global voice database

    Args:
        config: Config object
        asr_output_path: ASR result path
        event_path: event JSON path
        event_id: event ID
        video_path: optional video path (used to re-extract audio if missing)
    """
    print(f"\n{'='*70}")
    print(" Step 3: Voiceprint (speaker identification)")
    print(f"{'='*70}")

    # 1. Verify ASR result exists
    if not os.path.exists(asr_output_path):
        print(f"⚠️  Skipping: ASR result not found")
        return

    # 2. Load ASR result
    with open(asr_output_path, 'r', encoding='utf-8') as f:
        asr_result = json.load(f)

    if not asr_result.get("speech_segments"):
        print(f"⚠️  Skipping: no speech segments")
        return

    print(f"Loaded {len(asr_result['speech_segments'])} speech segments")

    # 3. Ensure audio is available
    temp_dir = Path(config.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_audio_path = temp_dir / f"{event_id}.wav"

    # Re-extract from video if temp audio is missing
    if not temp_audio_path.exists():
        original_audio_path = asr_result.get('audio_path', '')
        if original_audio_path and os.path.exists(original_audio_path):
            asr_result['audio_path'] = original_audio_path
        elif video_path and os.path.exists(video_path):
            print(f"Extracting audio from video...")
            extract_audio_from_video(video_path, str(temp_audio_path))
            asr_result['audio_path'] = str(temp_audio_path)
    else:
        asr_result['audio_path'] = str(temp_audio_path)

    # 4. Build VoiceprintProcessor
    voiceprint_processor = VoiceprintProcessor(config)

    # 5. Run voiceprint matching
    print(f"Matching voiceprints...")
    updated_event, voiceprint_db = voiceprint_processor.process_event(
        asr_result,
        asr_file_path=asr_output_path
    )

    # 5. Update speaker info in the event JSON
    if os.path.exists(event_path):
        with open(event_path, 'r', encoding='utf-8') as f:
            event_data = json.load(f)

        # Update speaker fields in interaction_language
        interaction_language = event_data.get("attributes", {}).get("interaction_language", [])
        if interaction_language:
            # Pull updated segments from updated_event
            updated_segments = updated_event.get("segments", [])

            # Build a (start, end) -> speaker map (high precision to avoid float-match issues)
            time_to_speaker = {}
            for seg in updated_segments:
                key = f"{seg['start_time']:.4f}-{seg['end_time']:.4f}"
                time_to_speaker[key] = seg.get('speaker')

            # Update speaker on each interaction_language entry
            for lang_item in interaction_language:
                start = lang_item.get('start_time', 0)
                end = lang_item.get('end_time', 0)
                key = f"{start:.4f}-{end:.4f}"
                if key in time_to_speaker:
                    lang_item['speaker'] = time_to_speaker[key]

            # Save updated event JSON
            with open(event_path, 'w', encoding='utf-8') as f:
                json.dump(event_data, f, ensure_ascii=False, indent=2)

    print(f"✓ Voiceprint done, identified {len(voiceprint_db)} speakers")
    for voice_record in voiceprint_db:
        voice_id = voice_record['voice_id']
        num_segments = len(voice_record['language'])
        print(f"  {voice_id}: {num_segments} segments")


def run_relation_step(config: Config, event_id: str, day: int):
    """
    Step 4: event-relation reasoning

    Pipeline:
    1. Find prior events within the time window
    2. Decide temporal relations (timestamp-based)
    3. Decide causal relations (LLM call)
    4. Update the event JSON `relations` field on both sides

    Args:
        config: Config object
        event_id: event ID
        day: day number
    """
    print(f"\n{'='*70}")
    print(" Step 4: Relation (event-relation reasoning)")
    print(f"{'='*70}")

    # Build processor
    processor = EventRelationProcessor(config)

    # Reason about relations
    result = processor.process_event(event_id, day)

    # Pull all valid relations (causal / activity + temporal)
    all_valid_relations = result.get('all_valid_relations', [])
    causal_activity_relations = result.get('causal_activity_relations', [])
    temporal_relations = result.get('temporal_relations', [])

    print(f"✓ Relation done")
    print(f"  - Causal/activity relations: {len(causal_activity_relations)}")
    print(f"  - Pure temporal relations:   {len(temporal_relations)}")
    print(f"  - Total:                     {len(all_valid_relations)}")

    # Print a short summary
    if all_valid_relations:
        print("\nRelation summary:")
        for rel in all_valid_relations[:5]:  # show first 5
            rel_type = rel['type']
            related_id = rel['related_event_id']
            reason = rel.get('reason', '')[:50]  # first 50 chars
            print(f"  [{rel_type}] -> {related_id}")
            if reason:
                print(f"      reason: {reason}...")
        if len(all_valid_relations) > 5:
            print(f"  ... and {len(all_valid_relations) - 5} more")


def main():
    parser = argparse.ArgumentParser(description='Process a single egocentric video through the multi-step pipeline.')
    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help='Path to the input video file.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to the pipeline config YAML.'
    )
    parser.add_argument(
        '--steps',
        type=str,
        nargs='+',
        default=None,
        help='Steps to run (asr, caption, voiceprint, relation, entity). Defaults to pipeline.steps in the config.'
    )

    args = parser.parse_args()

    # Validate video file
    if not os.path.exists(args.video):
        print(f"Error: video file not found: {args.video}")
        sys.exit(1)

    # Validate config file
    if not os.path.exists(args.config):
        print(f"Error: config file not found: {args.config}")
        sys.exit(1)

    print("="*70)
    print(" Egocentric Video Annotation Pipeline")
    print("="*70)
    print(f"Video: {args.video}")

    # Load config
    config = Config.from_yaml(args.config)
    config.ensure_dirs()

    # Decide which steps to run
    steps = args.steps if args.steps else config.pipeline_steps
    print(f"Steps: {' → '.join(steps)}")

    # Generate event_id
    video_info = parse_video_filename(args.video)
    event_id = generate_event_id(video_info["timestamp"])
    print(f"Event ID: {event_id}")

    # Track per-step intermediate paths
    asr_output_path = None
    event_path = None

    # Initialize the global object matcher (only when entity step is enabled and matching is on)
    global_matcher = None
    if "entity" in steps and config.entity_global_matching_enabled:
        print(f"\n[init] Building global object matcher")
        # GlobalObjectMatcher needs the full nested config dict
        global_matcher_config = config._config.get('entity', {})
        global_matcher = GlobalObjectMatcher(global_matcher_config)

    # Run each step in order
    for step in steps:
        step = step.lower()

        if step == "asr":
            asr_output_path = run_asr_step(config, args.video, event_id)

        elif step == "caption":
            event_path = run_caption_step(config, args.video, asr_output_path, event_id)

        elif step == "voiceprint":
            if not asr_output_path:
                # ASR step wasn't run this invocation; try to load an existing result
                asr_dir = Path(config.voiceprint_asr_dir)
                asr_output_path = str(asr_dir / f"{event_id}.json")

            if not event_path:
                # Caption step wasn't run this invocation; try to load existing event JSON
                day = video_info['day']
                event_path = str(Path(config.events_dir) / f"DAY{day}" / f"{event_id}.json")

            run_voiceprint_step(config, asr_output_path, event_path, event_id, video_path=args.video)

        elif step == "relation":
            # Event-relation reasoning
            if not event_path:
                # Caption step wasn't run this invocation; try to load existing event JSON
                day = video_info['day']
                event_path = str(Path(config.events_dir) / f"DAY{day}" / f"{event_id}.json")

            day = video_info['day']
            run_relation_step(config, event_id, day)

        elif step == "entity":
            # Object tracking + feature extraction
            if not event_path:
                # Caption step wasn't run this invocation; try to load existing event JSON
                day = video_info['day']
                event_path = str(Path(config.events_dir) / f"DAY{day}" / f"{event_id}.json")

            run_entity_step(config, event_path, global_matcher=global_matcher)

        else:
            print(f"⚠️  Unknown step: {step}")

    # Final summary
    print("\n" + "="*70)
    print(" Pipeline finished")
    print("="*70)

    if event_path and os.path.exists(event_path):
        with open(event_path, 'r', encoding='utf-8') as f:
            event_data = json.load(f)

        print(f"Caption: {event_data.get('caption')}")

        interaction_language = event_data.get('attributes', {}).get('interaction_language', [])
        if interaction_language:
            print(f"\nDialogue ({len(interaction_language)} entries):")
            for lang_item in interaction_language[:3]:
                speaker = lang_item.get('speaker', 'unknown')
                text = lang_item.get('text', '')
                print(f"  [{speaker}] {text[:60]}{'...' if len(text) > 60 else ''}")
            if len(interaction_language) > 3:
                print(f"  ...")

        # Object tracking summary
        interaction_objects = event_data.get('attributes', {}).get('interaction_object', [])
        if interaction_objects:
            print(f"\nObjects ({len(interaction_objects)}):")
            for obj in interaction_objects[:5]:
                obj_name = obj.get('name', 'unknown')
                has_tracking = 'tracking_segments' in obj
                has_features = 'features' in obj
                global_name = obj.get('global_object_name', '')

                info_parts = [obj_name]
                if has_tracking:
                    segments = obj.get('tracking_segments', [])
                    total_frames = sum(len(seg.get('bboxes', [])) for seg in segments)
                    info_parts.append(f"track:{total_frames} frames")
                if has_features:
                    info_parts.append("features:✓")
                if global_name:
                    info_parts.append(f"global:{global_name}")

                print(f"  - {' | '.join(info_parts)}")
            if len(interaction_objects) > 5:
                print(f"  ...")

    print("="*70)

    # Clean up temp audio file
    temp_dir = Path(config.temp_dir)
    temp_audio_path = temp_dir / f"{event_id}.wav"
    if temp_audio_path.exists():
        try:
            os.remove(temp_audio_path)
        except Exception:
            pass

    sys.exit(0)


if __name__ == "__main__":
    main()
