#!/usr/bin/env python3
"""
Visualize the per-object bbox tracks stored in an event JSON.

Usage:
    python visualize_bbox.py --event_json <event JSON path> [--output_dir <output dir>] [--fps 5]

Examples:
    python visualize_bbox.py --event_json ../data/events/DAY1/DAY1_11100000_evt.json
    python visualize_bbox.py --event_json ../data/events/DAY1/DAY1_11100000_evt.json --output_dir ./vis_output --fps 10
"""

import json
import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import List, Tuple, Dict
import colorsys


def generate_colors(n: int) -> List[Tuple[int, int, int]]:
    """Generate n visually distinct BGR colors."""
    colors = []
    for i in range(n):
        hue = i / n
        # High saturation + value for visibility
        rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        colors.append(bgr)
    return colors


def resize_frame_to_640(frame: np.ndarray, target_long_side: int = 640) -> Tuple[np.ndarray, float]:
    """
    Resize a frame so its long side equals 640.

    Returns:
        resized_frame: the resized frame
        scale: applied scale factor
    """
    h, w = frame.shape[:2]
    if h > w:
        new_h = target_long_side
        new_w = int(w * (target_long_side / h))
        scale = target_long_side / h
    else:
        new_w = target_long_side
        new_h = int(h * (target_long_side / w))
        scale = target_long_side / w

    resized_frame = cv2.resize(frame, (new_w, new_h))
    return resized_frame, scale


def draw_bbox_on_frame(frame: np.ndarray, bbox: List[float], color: Tuple[int, int, int],
                       label: str, thickness: int = 2) -> np.ndarray:
    """
    Draw a bbox on the frame.

    Args:
        frame: input frame
        bbox: [x1, y1, x2, y2] in pixel coordinates of the resized frame
        color: BGR color
        label: text label
        thickness: rectangle line thickness
    """
    x1, y1, x2, y2 = [int(coord) for coord in bbox]

    # Draw rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Draw label background
    label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y1_label = max(y1, label_size[1] + 10)
    cv2.rectangle(frame, (x1, y1_label - label_size[1] - 10),
                  (x1 + label_size[0], y1_label), color, -1)

    # Draw label text
    cv2.putText(frame, label, (x1, y1_label - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


def visualize_event_bbox(event_json_path: str, output_dir: str = None, fps: int = 5):
    """
    Visualize the bbox tracks stored in an event JSON.

    Args:
        event_json_path: path to the event JSON
        output_dir: output directory; defaults to vis_<event_id>/ next to the JSON
        fps: output video frame rate
    """
    # Load event JSON
    event_json_path = Path(event_json_path)
    if not event_json_path.exists():
        raise FileNotFoundError(f"Event JSON not found: {event_json_path}")

    with open(event_json_path, 'r', encoding='utf-8') as f:
        event_data = json.load(f)

    event_id = event_data['event_id']
    video_path = event_data['video_path']
    interaction_objects = event_data['attributes']['interaction_object']

    print(f"Event ID: {event_id}")
    print(f"Video path: {video_path}")
    print(f"Interaction objects: {len(interaction_objects)}")

    # Verify video exists
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Build output directory
    if output_dir is None:
        output_dir = event_json_path.parent / f"vis_{event_id}"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video info: {video_width}x{video_height}, {video_fps} fps, {total_frames} frames")

    # Assign a color per object
    colors = generate_colors(len(interaction_objects))

    # Build a bbox index: {timestamp: {object_name: bbox}}
    bbox_index = {}
    for obj_idx, obj in enumerate(interaction_objects):
        obj_name = obj['name']
        obj_description = obj['description']

        if 'tracking_segments' not in obj or len(obj['tracking_segments']) == 0:
            print(f"Object {obj_name} has no tracking data, skipping")
            continue

        for segment in obj['tracking_segments']:
            bboxes = segment['bboxes']
            timesteps = segment['timesteps']

            for bbox, timestep in zip(bboxes, timesteps):
                if timestep not in bbox_index:
                    bbox_index[timestep] = {}

                bbox_index[timestep][obj_name] = {
                    'bbox': bbox,
                    'color': colors[obj_idx],
                    'description': obj_description
                }

    print(f"Bbox index built, {len(bbox_index)} timestamps")

    # Prepare output video
    output_video_path = output_dir / f"{event_id}_bbox_vis.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Read first frame to determine output size
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read the first video frame")

    resized_first_frame, scale = resize_frame_to_640(first_frame)
    out_height, out_width = resized_first_frame.shape[:2]

    print(f"Output size: {out_width}x{out_height} (scale: {scale:.3f})")

    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (out_width, out_height))

    # Reset to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Process each frame
    frame_idx = 0
    frames_with_bbox = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Current timestamp (seconds)
        current_time = frame_idx / video_fps

        # Resize to 640
        resized_frame, _ = resize_frame_to_640(frame)

        # Draw bboxes
        has_bbox = False
        if current_time in bbox_index:
            has_bbox = True
            for obj_name, obj_info in bbox_index[current_time].items():
                bbox = obj_info['bbox']
                color = obj_info['color']
                description = obj_info['description']

                # Label = object name
                label = f"{obj_name}"

                resized_frame = draw_bbox_on_frame(resized_frame, bbox, color, label)

        if has_bbox:
            frames_with_bbox += 1

        # Top-left timestamp + frame info
        info_text = f"Time: {current_time:.2f}s | Frame: {frame_idx}/{total_frames}"
        cv2.putText(resized_frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Write to output
        out.write(resized_frame)

        frame_idx += 1

        # Log progress every 100 frames
        if frame_idx % 100 == 0:
            print(f"Progress: {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.1f}%)")

    # Release resources
    cap.release()
    out.release()

    print(f"\nVisualization done.")
    print(f"Total frames: {frame_idx}")
    print(f"Frames with bbox: {frames_with_bbox}")
    print(f"Output video: {output_video_path}")

    # Save object info to a text file
    info_file = output_dir / f"{event_id}_objects_info.txt"
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write(f"Event ID: {event_id}\n")
        f.write(f"Video path: {video_path}\n")
        f.write(f"Caption: {event_data.get('caption', 'N/A')}\n\n")
        f.write(f"Interaction objects: {len(interaction_objects)}\n\n")

        for idx, obj in enumerate(interaction_objects):
            f.write(f"Object {idx + 1}: {obj['name']}\n")
            f.write(f"  Color: RGB{colors[idx][::-1]}\n")  # convert BGR to RGB for display
            f.write(f"  Description: {obj['description']}\n")
            f.write(f"  Action: {obj['action']}\n")
            f.write(f"  Location: {obj['location']}\n")

            if 'global_object_name' in obj:
                f.write(f"  Global object name: {obj['global_object_name']}\n")

            if 'tracking_segments' in obj:
                total_bboxes = sum(len(seg['bboxes']) for seg in obj['tracking_segments'])
                f.write(f"  Tracking segments: {len(obj['tracking_segments'])}\n")
                f.write(f"  Total bboxes: {total_bboxes}\n")

                for seg_idx, seg in enumerate(obj['tracking_segments']):
                    f.write(f"    Segment {seg_idx + 1}: {seg['segment_start_time']:.1f}s - {seg['segment_end_time']:.1f}s ({len(seg['bboxes'])} bboxes)\n")

            f.write("\n")

    print(f"Object info: {info_file}")


def main():
    parser = argparse.ArgumentParser(description="Visualize the per-object bbox tracks stored in an event JSON.")
    parser.add_argument('--event_json', type=str, required=True,
                        help='Path to the event JSON file.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: vis_<event_id>/ next to the JSON file).')
    parser.add_argument('--fps', type=int, default=5,
                        help='Output video frame rate (default: 5).')

    args = parser.parse_args()

    try:
        visualize_event_bbox(args.event_json, args.output_dir, args.fps)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
