# Event-Based Camera Visualizer

A Python-based visualization tool for event camera data streams, designed to process and animate neuromorphic vision sensor outputs.

## Overview

This project processes event data from event-based cameras (tested with N-MNIST dataset) and creates animated visualizations showing temporal event accumulation. Event cameras capture per-pixel brightness changes asynchronously, producing sparse event streams instead of traditional frames.

## Example Output

![Event Animation](event_animation.gif)

## Features

- **Adaptive Frame Generation**: Automatically calculates optimal events per frame based on event rate
- **Configurable Intervals**: Adjust temporal grouping (default: 50ms intervals)
- **Privacy Preservation**: Optional shuffling of events within chunks or frame reordering
- **Polarity Visualization**: Color-coded display of positive/negative events (red/blue)
- **GIF Export**: Generates animated visualizations for analysis

## Requirements

```bash
pip install numpy matplotlib
```

## Usage

1. Place your event data in `events.txt` (format: timestamp, x, y, polarity)
2. Run the visualizer:
```bash
python visualizer.py
```

The script will:
- Load and process event data
- Calculate optimal chunk sizes based on event rate
- Generate frames (240×180 resolution)
- Create `event_animation.gif` output
- Display interactive animation

## Configuration

Edit variables in `visualizer.py`:
- `interval`: Time window in milliseconds (default: 50)
- `width`, `height`: Output resolution (default: 240×180)
- Uncomment shuffle functions for privacy-preserving mode

## Data Format

Input file should contain one event per line:
```
timestamp x_coordinate y_coordinate polarity
```
- Timestamp: microseconds or seconds
- Coordinates: integer pixel positions
- Polarity: 1 (brightness increase) or 0/-1 (brightness decrease)
