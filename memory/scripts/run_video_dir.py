#!/usr/bin/env python3
"""
Batch video-directory processing script with multi-step pipeline support.

Supported steps:
1. asr: speech recognition (extract audio from the video and run ASR)
2. caption: event annotation (generate the event description)
3. voiceprint: speaker identification (match speakers against a global DB)
4. relation: event-relation reasoning (temporal / causal links to prior events)
5. entity: object tracking + global matching (detect, track interaction targets,
           and match entities across events)

Usage:
python scripts/run_video_dir.py \
  --video_dir /path/to/video_dir \
  --config configs/config.yaml \
  --start_idx 0 \
  --num_videos 10

Notes:
- Auto-scans the directory for .mp4 files
- Sorts by filename so timestamps are processed in chronological order
- Supports start_idx / num_videos slicing
- Models are initialized once and reused across all videos (much faster batch runs)
"""

import argparse
import sys
import os
import json
import time
from pathlib import Path
from tqdm import tqdm

# Add src/ to PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import Config
from event import EventAnnotator
from event.event_relation_processor import EventRelationProcessor
from event.event_storage import EventStorage
from voice.asr import create_asr_processor
from voice.voiceprint import VoiceprintProcessor
from utils.audio_utils import extract_audio_from_video
from utils import parse_video_filename, generate_event_id
from entity.object_tracker import ObjectTracker
from entity.global_entity_manager import GlobalEntityManager


def run_asr_step(config: Config, video_path: str, event_id: str, asr_processor) -> tuple:
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
        asr_processor: shared ASR processor instance (reused across videos)

    Returns:
        (asr_output_path, elapsed_seconds)
    """
    start_time = time.time()
    print(f"\n{'='*70}")
    print(" Step 1: ASR (speech recognition)")
    print(f"{'='*70}")

    # 1. Prepare output path
    asr_dir = Path(config.voiceprint_asr_dir)
    asr_dir.mkdir(parents=True, exist_ok=True)
    asr_output_path = asr_dir / f"{event_id}.json"

    if os.path.exists(asr_output_path):
        elapsed = time.time() - start_time
        print(f"✓ ASR result already exists, skipping ({elapsed:.2f}s)")
        return str(asr_output_path), elapsed

    # 2. Extract audio to a temp directory
    temp_dir = Path(config.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_audio_path = temp_dir / f"{event_id}.wav"

    print(f"Extracting audio...")
    extract_audio_from_video(video_path, str(temp_audio_path))

    # 3. Run ASR (reusing the shared processor)
    print(f"Running ASR...")
    asr_result = asr_processor.process(
        audio_path=str(temp_audio_path),
        event_id=event_id
    )

    # 5. Save result (custom format collapses speech_segments to one line per segment)
    event_storage = EventStorage(config.events_dir, config.features_dir)
    json_str = event_storage._custom_json_format(asr_result)
    with open(asr_output_path, 'w', encoding='utf-8') as f:
        f.write(json_str)

    elapsed = time.time() - start_time
    print(f"✓ ASR done, found {len(asr_result['speech_segments'])} speech segments ({elapsed:.2f}s)")

    # Note: keep the temp audio for the voiceprint step; cleaned up at the end of each video.

    return str(asr_output_path), elapsed


def run_caption_step(config: Config, video_path: str, asr_output_path: str, event_id: str, annotator: EventAnnotator) -> tuple:
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
        annotator: shared EventAnnotator instance (reused across videos)

    Returns:
        (event_path, elapsed_seconds)
    """
    start_time = time.time()
    print(f"\n{'='*70}")
    print(" Step 2: Caption (event annotation)")
    print(f"{'='*70}")

    # 0. Skip if event JSON already exists
    video_info = parse_video_filename(video_path)
    day = video_info['day']
    event_path = Path(config.events_dir) / f"DAY{day}" / f"{event_id}.json"

    if os.path.exists(event_path):
        elapsed = time.time() - start_time
        print(f"✓ Event already exists, skipping ({elapsed:.2f}s)")
        return str(event_path), elapsed

    # 1. Load ASR result
    speech_segments = None
    if asr_output_path and os.path.exists(asr_output_path):
        with open(asr_output_path, 'r', encoding='utf-8') as f:
            asr_result = json.load(f)
            speech_segments = asr_result.get("speech_segments")
        print(f"Loaded {len(speech_segments) if speech_segments else 0} speech segments")

    # 2. Annotate (reusing the shared annotator)
    print(f"Generating event annotation...")
    try:
        event_ids = annotator.annotate_video(video_path, speech_segments_json=speech_segments)

        # 4. Resolve event JSON path
        actual_event_id = event_ids[0] if event_ids else event_id
        video_info = parse_video_filename(video_path)
        day = video_info['day']
        event_path = Path(config.events_dir) / f"DAY{day}" / f"{actual_event_id}.json"

        elapsed = time.time() - start_time
        print(f"✓ Caption done ({elapsed:.2f}s)")

        return str(event_path), elapsed

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Caption failed, skipping this video (error: {str(e)[:100]})")
        return None, elapsed


def run_voiceprint_step(config: Config, event_path: str, event_id: str, voiceprint_processor: VoiceprintProcessor, video_path: str = None) -> float:
    """
    Step 3: Voiceprint (speaker identification)

    Pipeline:
    1. Call VoiceprintProcessor.process() (new implementation)
    2. Internally: extract audio, match voiceprints, write back to event JSON,
       update the global voice DB
    3. Does not save merged audio and does not reset the voice DB

    Args:
        config: Config object
        event_path: event JSON path
        event_id: event ID (kept for API compatibility, unused)
        voiceprint_processor: shared VoiceprintProcessor instance (reused across videos)
        video_path: optional video path (kept for API compatibility, unused)

    Returns:
        Elapsed seconds.
    """
    start_time = time.time()
    print(f"\n{'='*70}")
    print(" Step 3: Voiceprint (speaker identification)")
    print(f"{'='*70}")

    # Verify event JSON exists
    if not os.path.exists(event_path):
        elapsed = time.time() - start_time
        print(f"⚠️  Skipping: event JSON not found ({elapsed:.2f}s)")
        return elapsed

    # Skip if already processed (interaction_language already has global_voice fields)
    # AND those voice_ids actually exist in the database
    with open(event_path, 'r', encoding='utf-8') as f:
        event_data = json.load(f)

    interaction_language = event_data.get('attributes', {}).get('interaction_language', [])
    if interaction_language and any('global_voice' in seg for seg in interaction_language):
        # Collect referenced voice_ids
        voice_ids_in_event = set()
        for seg in interaction_language:
            if 'global_voice' in seg:
                voice_ids_in_event.add(seg['global_voice'])

        # Check those voice_ids exist in the database
        database_voices = voiceprint_processor.database.get_all_voices()
        voice_ids_in_db = set(v['voice_id'] for v in database_voices)

        # Skip only if all referenced voice_ids are in the database
        if voice_ids_in_event.issubset(voice_ids_in_db):
            elapsed = time.time() - start_time
            print(f"✓ Voiceprint already processed and DB intact, skipping ({elapsed:.2f}s)")
            return elapsed
        else:
            missing_voices = voice_ids_in_event - voice_ids_in_db
            print(f"⚠️  Event has global_voice but DB is missing {missing_voices}; reprocessing...")

    # Run the new voiceprint matching pipeline
    # save_merged_audio=False: don't save merged audio (temp files cleaned automatically)
    # write_back=True: write updated speaker fields back into the event JSON
    result = voiceprint_processor.process(
        event_json_path=event_path,
        save_merged_audio=False,
        write_back=True
    )

    # Print summary
    voice_mapping = result.get('voice_mapping', {})
    elapsed = time.time() - start_time
    print(f"✓ Voiceprint done, matched {len(voice_mapping)} speakers ({elapsed:.2f}s)")
    for speaker, voice_id in voice_mapping.items():
        print(f"  {speaker} → {voice_id}")

    return elapsed


def run_entity_step(config: Config, event_path: str, object_tracker: ObjectTracker,
                    global_entity_manager: GlobalEntityManager = None) -> float:
    """
    Step 5: Entity (object tracking + global matching)

    Pipeline:
    1. Call ObjectTracker.process_event() to detect and track objects
    2. If global matching is enabled, call GlobalEntityManager.process_event() to
       match entities across events
    3. The interaction_object field in the event JSON is updated in place
       (bbox, timestep, ...)

    Args:
        config: Config object
        event_path: event JSON path
        object_tracker: shared ObjectTracker instance (reused across videos)
        global_entity_manager: optional shared GlobalEntityManager instance

    Returns:
        Elapsed seconds.
    """
    start_time = time.time()
    print(f"\n{'='*70}")
    print(" Step 5: Entity (object tracking + global matching)")
    print(f"{'='*70}")

    # Verify event JSON exists
    if not os.path.exists(event_path):
        elapsed = time.time() - start_time
        print(f"⚠️  Skipping: event JSON not found ({elapsed:.2f}s)")
        return elapsed

    # Skip if already processed (every interaction_object has bbox + global_name)
    with open(event_path, 'r', encoding='utf-8') as f:
        event_data = json.load(f)

    interaction_objects = event_data.get('attributes', {}).get('interaction_object', [])
    if interaction_objects and all('bbox' in obj and 'global_name' in obj for obj in interaction_objects):
        elapsed = time.time() - start_time
        print(f"✓ Entity already processed, skipping ({elapsed:.2f}s)")
        return elapsed

    # Substep 1: object tracking (detect, track, save crops)
    # save_entity=True: save tracking data to entities/event/{event_id}/{object_name}.json
    # update_event=True: write bbox/timestep/... back into the event JSON's interaction_object
    print(f"Running object tracking...")
    entity_paths = object_tracker.process_event(
        event_json_path=event_path,
        save_entity=True,
        update_event=True,
        visualize=False
    )

    if not entity_paths:
        elapsed = time.time() - start_time
        print(f"✓ Entity done, no interaction objects ({elapsed:.2f}s)")
        return elapsed

    print(f"✓ Object tracking done, generated {len(entity_paths)} entity files")

    # Substep 2: global entity matching (if enabled)
    if global_entity_manager:
        print(f"Running global entity matching...")
        global_entity_manager.process_event(
            event_json_path=event_path,
            output_path=event_path  # overwrite original to write global_name back
        )
        elapsed = time.time() - start_time
        print(f"✓ Global matching done ({elapsed:.2f}s)")
    else:
        elapsed = time.time() - start_time
        print(f"✓ Entity done (global matching disabled) ({elapsed:.2f}s)")

    return elapsed


def run_relation_step(config: Config, event_id: str, day: int, relation_processor: EventRelationProcessor) -> float:
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
        relation_processor: shared EventRelationProcessor instance (reused across videos)

    Returns:
        Elapsed seconds.
    """
    start_time = time.time()
    print(f"\n{'='*70}")
    print(" Step 4: Relation (event-relation reasoning)")
    print(f"{'='*70}")

    # Skip if already processed (event JSON already has a `relations` field)
    event_path = str(Path(config.events_dir) / f"DAY{day}" / f"{event_id}.json")
    if os.path.exists(event_path):
        with open(event_path, 'r', encoding='utf-8') as f:
            event_data = json.load(f)

        if 'relations' in event_data and event_data['relations']:
            elapsed = time.time() - start_time
            print(f"✓ Relation already processed, skipping ({elapsed:.2f}s)")
            return elapsed

    # Reason (reusing the shared processor)
    result = relation_processor.process_event(event_id, day)

    # Pull all valid relations (causal / activity + temporal)
    all_valid_relations = result.get('all_valid_relations', [])
    causal_activity_relations = result.get('causal_activity_relations', [])
    temporal_relations = result.get('temporal_relations', [])

    elapsed = time.time() - start_time
    print(f"✓ Relation done ({elapsed:.2f}s)")
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

    return elapsed


def scan_and_sort_videos(video_dir: str) -> list:
    """
    Scan a directory for video files and sort by filename timestamp.

    Behavior:
    1. Lists every .mp4 file in the directory
    2. Sorts by filename lexicographically (filenames like DAY1_11113000.mp4 sort
       in chronological order naturally)

    Args:
        video_dir: video directory path

    Returns:
        Sorted list of video paths (as strings).
    """
    video_dir = Path(video_dir)
    # List all .mp4 files
    video_files = list(video_dir.glob("*.mp4"))

    # Sort by filename (timestamp embedded in name → lexicographic == chronological)
    video_files.sort(key=lambda x: x.name)

    return [str(v) for v in video_files]


def main():
    parser = argparse.ArgumentParser(description='Batch-process a directory of egocentric videos through the multi-step pipeline.')
    parser.add_argument(
        '--video_dir',
        type=str,
        required=True,
        help='Path to the directory containing input videos.'
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
    parser.add_argument(
        '--start_idx',
        type=int,
        default=0,
        help='Index of the first video to process (default: 0).'
    )
    parser.add_argument(
        '--num_videos',
        type=int,
        default=None,
        help='Number of videos to process (default: process to end of directory).'
    )

    args = parser.parse_args()

    # Validate video directory
    if not os.path.exists(args.video_dir):
        print(f"Error: video directory not found: {args.video_dir}")
        sys.exit(1)

    # Validate config file
    if not os.path.exists(args.config):
        print(f"Error: config file not found: {args.config}")
        sys.exit(1)

    print("="*70)
    print(" Egocentric Video Batch Annotation Pipeline")
    print("="*70)
    print(f"Video directory: {args.video_dir}")

    # Load config
    config = Config.from_yaml(args.config)
    config.ensure_dirs()

    # Scan + sort videos
    print(f"Scanning video files...")
    all_videos = scan_and_sort_videos(args.video_dir)
    print(f"Found {len(all_videos)} video files")

    # Slice based on start_idx / num_videos
    # e.g. start_idx=10, num_videos=5 -> process videos[10:15]
    end_idx = args.start_idx + args.num_videos if args.num_videos else len(all_videos)
    videos_to_process = all_videos[args.start_idx:end_idx]
    print(f"Processing range: [{args.start_idx}:{end_idx}], total {len(videos_to_process)} videos")

    # Decide which steps to run
    steps = args.steps if args.steps else config.pipeline_steps
    print(f"Steps: {' → '.join(steps)}")

    # ===== Model initialization (once, reused across all videos) =====
    # Key optimization: hoist all model loads out of the loop so each video
    # doesn't pay the model-init cost. This dramatically speeds up batch runs.
    print(f"\n{'='*70}")
    print(" Initializing models (reused across all videos)")
    print(f"{'='*70}")

    # ASR processor (loads Whisper / Gemini)
    asr_processor = None
    if "asr" in steps:
        asr_processor = create_asr_processor(config, asr_type=config.asr_client)
        print(f"✓ ASR processor ready")

    # EventAnnotator (loads GPT-4o / Gemini client + embedding model)
    annotator = None
    if "caption" in steps:
        annotator = EventAnnotator(config)
        print(f"✓ EventAnnotator ready")

    # EventRelationProcessor (loads the configured LLM client)
    relation_processor = None
    if "relation" in steps:
        relation_processor = EventRelationProcessor(config)
        print(f"✓ EventRelationProcessor ready")

    # ObjectTracker (loads Grounding DINO + SAM2 — slow init)
    object_tracker = None
    global_entity_manager = None
    if "entity" in steps:
        # Build the object tracker
        object_tracker = ObjectTracker(config)
        print(f"✓ ObjectTracker ready (Grounding DINO + SAM2)")

        # Build the global entity manager (cross-event matching)
        if config.entity_global_matching_enabled:
            global_entity_manager = GlobalEntityManager(config, visualize=False)
            print(f"✓ GlobalEntityManager ready")

            # Reset the global entity DB at the start of the batch if requested
            reset_flag = config._config["entity"]["global_matching"].get("reset_global_entities", False)
            if reset_flag:
                global_entity_manager.reset_global()
                print(f"  Global entity DB reset (reset_global_entities=True)")

    # VoiceprintProcessor (loads voiceprint model + voice DB)
    voiceprint_processor = None
    if "voiceprint" in steps:
        voiceprint_processor = VoiceprintProcessor(config)
        print(f"✓ VoiceprintProcessor ready")

    # ===== Main loop: one video at a time =====
    print(f"\n{'='*70}")
    print(" Starting batch run")
    print(f"{'='*70}")

    # Stats
    step_times = {step: [] for step in steps}  # per-step elapsed times
    video_times = []  # per-video elapsed times

    # tqdm progress bar
    pbar = tqdm(enumerate(videos_to_process, start=args.start_idx),
                total=len(videos_to_process),
                desc="Processing videos",
                unit="video")

    for video_idx, video_path in pbar:
        video_start_time = time.time()

        print(f"\n{'#'*70}")
        print(f" Video [{video_idx + 1}/{len(all_videos)}]: {Path(video_path).name}")
        print(f"{'#'*70}")

        # Build event_id and pull basic info from filename
        video_info = parse_video_filename(video_path)
        event_id = generate_event_id(video_info["timestamp"])
        day = video_info['day']
        print(f"Event ID: {event_id}")

        # Per-video intermediate paths
        asr_output_path = None
        event_path = None

        # Run each configured step in order
        for step in steps:
            step = step.lower()
            step_time = 0

            if step == "asr":
                asr_output_path, step_time = run_asr_step(config, video_path, event_id, asr_processor)

            elif step == "caption":
                event_path, step_time = run_caption_step(config, video_path, asr_output_path, event_id, annotator)
                # If Caption failed, skip the rest of the pipeline for this video
                if event_path is None:
                    print(f"⚠️  Caption failed; skipping remaining steps for this video")
                    break

            elif step == "relation":
                # Event-relation reasoning
                if not event_path:
                    # Caption step wasn't run this invocation; try to load existing event JSON
                    event_path = str(Path(config.events_dir) / f"DAY{day}" / f"{event_id}.json")

                step_time = run_relation_step(config, event_id, day, relation_processor)

            elif step == "entity":
                # Object tracking + global matching
                if not event_path:
                    # Caption step wasn't run this invocation; try to load existing event JSON
                    event_path = str(Path(config.events_dir) / f"DAY{day}" / f"{event_id}.json")

                step_time = run_entity_step(config, event_path, object_tracker, global_entity_manager)

            elif step == "voiceprint":
                # Voiceprint (must come after entity so global_name is available)
                if not event_path:
                    # Caption step wasn't run this invocation; try to load existing event JSON
                    event_path = str(Path(config.events_dir) / f"DAY{day}" / f"{event_id}.json")

                step_time = run_voiceprint_step(config, event_path, event_id, voiceprint_processor, video_path=video_path)

            else:
                print(f"⚠️  Unknown step: {step}")

            # Record per-step elapsed
            if step_time:
                step_times[step].append(step_time)

        # Clean up the per-video temp audio
        temp_dir = Path(config.temp_dir)
        temp_audio_path = temp_dir / f"{event_id}.wav"
        if temp_audio_path.exists():
            os.remove(temp_audio_path)

        # Record per-video elapsed
        video_elapsed = time.time() - video_start_time
        video_times.append(video_elapsed)

        # Update tqdm postfix
        pbar.set_postfix({"current": f"{video_elapsed:.1f}s", "avg": f"{sum(video_times)/len(video_times):.1f}s"})

    # Final batch stats
    print("\n" + "="*70)
    print(" Batch run finished")
    print("="*70)
    print(f"Processed {len(videos_to_process)} videos")
    print(f"Index range: [{args.start_idx}:{end_idx}]")
    print(f"Total elapsed: {sum(video_times):.2f}s ({sum(video_times)/60:.2f} min)")
    print(f"Average per video: {sum(video_times)/len(video_times):.2f}s")
    print("\nPer-step stats:")
    for step in steps:
        if step_times[step]:
            avg_time = sum(step_times[step]) / len(step_times[step])
            total_time = sum(step_times[step])
            print(f"  {step:12s}: avg {avg_time:6.2f}s/video, total {total_time:7.2f}s ({total_time/60:.2f} min)")
    print("="*70)

    sys.exit(0)


if __name__ == "__main__":
    main()
