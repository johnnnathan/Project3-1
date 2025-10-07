# Performance Optimization Research

## Current Bottlenecks Analysis

Based on our current implementation in `visualizer.py`:
- Loading entire 510MB event file into memory at once (`np.loadtxt`)
- Creating full frames array in memory before processing
- Sequential frame generation (loop through all events)
- No parallelization or GPU acceleration

## Recommended Optimization Strategies

### 1. Memory-Efficient Data Loading

**Problem**: `np.loadtxt()` loads entire file into memory
**Solutions**:
- Use `np.memmap` for memory-mapped file access (access as if in memory, stored on disk)
- Chunk-based loading with `pd.read_csv(chunksize=...)`
- Use NPZ compressed format instead of text files

```python
# Memory-mapped approach
events_mmap = np.memmap('events.npy', dtype='float32', mode='r', shape=(n, 4))

# Or convert to NPZ format once
np.savez_compressed('events.npz', data=events)
events = np.load('events.npz')['data']
```

### 2. Vectorization Improvements

**Current**: Loop creates frames sequentially
**Optimization**: Use advanced NumPy indexing

```python
# Instead of loop, use np.add.at for accumulation
def create_frames_vectorized(x, y, p, chunk_size, n, height, width):
    n_frames = (n + chunk_size - 1) // chunk_size
    frames = np.zeros((n_frames, height, width))
    frame_indices = np.arange(n) // chunk_size
    np.add.at(frames, (frame_indices, y, x), p)
    return frames
```

### 3. Data Type Optimization

**Current**: Using default float64
**Optimization**: Use smaller dtypes

```python
# Use float32 instead of float64 (half the memory)
events = np.loadtxt("events.txt", dtype=np.float32)

# For coordinates, use int16 if resolution < 32768
x = events[:, 1].astype(np.int16)
y = events[:, 2].astype(np.int16)

# For polarity, use int8 (-1, 1)
p = np.where(events[:, 3]==1, 1, -1).astype(np.int8)
```

### 4. Parallel Processing

**Options**:
- **Numba JIT**: Just-in-time compilation for hot loops
- **Dask**: Parallel processing for chunked operations
- **multiprocessing**: Parallelize frame generation

```python
from numba import jit

@jit(nopython=True)
def create_frame_numba(x, y, p, height, width):
    frame = np.zeros((height, width))
    for i in range(len(x)):
        frame[y[i], x[i]] += p[i]
    return frame
```

### 5. GPU Acceleration (for large datasets)

**CuPy**: Drop-in NumPy replacement for NVIDIA GPUs

```python
import cupy as cp

# Move data to GPU
events_gpu = cp.asarray(events)
# All NumPy operations now run on GPU
```

### 6. Streaming/Online Processing

For very long videos, process in temporal windows:

```python
def process_streaming(event_file, window_size, overlap):
    # Process overlapping temporal windows
    # Only keep active window in memory
    # Write frames incrementally to video file
    pass
```

## Recommended Implementation Priority

1. **Quick wins** (implement first):
   - Switch to float32/int16 data types → 50% memory reduction
   - Vectorize frame creation → 5-10x speedup
   - Save/load as NPZ compressed → faster I/O

2. **Medium effort** (next phase):
   - Implement memory-mapped file access
   - Add Numba JIT for hot loops
   - Chunk-based processing for temporal streaming

3. **Advanced** (for production/large scale):
   - GPU acceleration with CuPy
   - Parallel processing with Dask
   - Custom C++ extensions with Cython

## Benchmarking Plan

Create test suite with varying:
- Image dimensions: 240×180, 640×480, 1280×720, 1920×1080
- Dataset sizes: 1M, 10M, 100M, 1B events
- Measure: memory usage, processing time, frame generation rate
