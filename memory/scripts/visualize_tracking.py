#!/usr/bin/env python3
"""
Standalone visualization script: render a tracking JSON as a video.
"""

import sys
import os

# Add project root to PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.entity.vis import EntityVisualizer


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize object-tracking results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("tracked_json", help="Path to the tracking-result JSON.")
    parser.add_argument("-o", "--output-dir", default=None,
                       help="Output directory (default: same as the JSON file).")
    parser.add_argument("-n", "--output-name", default=None,
                       help="Output filename (default: auto-generated).")
    parser.add_argument("--no-description", action="store_true",
                       help="Hide object descriptions.")
    parser.add_argument("--show-confidence", action="store_true",
                       help="Show detection confidence.")
    parser.add_argument("--bbox-thickness", type=int, default=2,
                       help="Bbox line thickness.")
    parser.add_argument("--font-scale", type=float, default=0.6,
                       help="Font scale.")

    args = parser.parse_args()

    # Build visualizer
    visualizer = EntityVisualizer(
        bbox_thickness=args.bbox_thickness,
        font_scale=args.font_scale
    )

    # Run visualization
    output_path = visualizer.visualize_from_command_line(
        tracked_json_path=args.tracked_json,
        output_dir=args.output_dir,
        output_name=args.output_name,
        show_description=not args.no_description,
        show_confidence=args.show_confidence
    )

    print(f"\n✓ Visualization done. Output: {output_path}")


if __name__ == "__main__":
    main()
