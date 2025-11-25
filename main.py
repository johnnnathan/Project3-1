import argparse
from loaders.txt_loader import load_txt
from loaders.npz_loader import load_npz
from processor.aggregator import aggregate_events_to_frames
from renderer.gif_renderer import save_gif

def main():
    parser = argparse.ArgumentParser(
        description="Event-based camera visualizer"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input file (.txt or .npz)"
    )
    parser.add_argument(
        "--format",
        choices=["txt", "npz"],
        default="txt",
        help="Input file format"
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Time window size for aggregation"
    )
    parser.add_argument(
        "--output",
        default="out.gif",
        help="Output GIF file path"
    )

    args = parser.parse_args()

    # Select loader
    if args.format == "txt":
        events = load_txt(args.input)
    else:
        events = load_npz(args.input)

    # Frame generation
    frames, timestamps = aggregate_events_to_frames(
        events,
        resolution=(180, 240),
        dt=args.dt
    )

    # Save GIF
    save_gif(frames, args.output)

    print(f"Generated {len(frames)} frames")
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
