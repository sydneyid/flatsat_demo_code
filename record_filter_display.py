#!/usr/bin/env python3
"""
Record from a Metavision event camera: run a short calibration (0.1 s) to identify
hot pixels, mask them at hardware (if supported), then record 10 s with hot pixels
already excluded. Apply activity (500 ms, 10x10) and neighborhood (250 ms, 3x3)
filters, write to RAW, render to MP4, and display.

Requires: Metavision SDK Python (metavision_core, metavision_sdk_core).
Optional: scipy for fast cKDTree-based activity/neighborhood filters; numba as fallback; opencv-python for MP4. No live preview.
"""

import argparse
import os
import sys
import time

import numpy as np

try:
    from scipy.spatial import cKDTree
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def dec(f):
            return f
        return dec if not args else dec(args[0])

# Metavision SDK
from metavision_core.event_io import EventsIterator
from metavision_core.event_io.raw_reader import initiate_device
from metavision_sdk_base import EventCDBuffer
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_stream import RAWEvt2EventFileWriter

# Video: prefer OpenCV for MP4 write/display; fallback to skvideo if needed
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

RECORD_DURATION_S = 10
CALIBRATION_DURATION_S = 0.1  # Short recording to identify hot pixels before main recording
HOT_PIXEL_TOP_K = 10
FRAME_FPS = 25
DELTA_T_US = 10000  # 10 ms slices for iteration

# Activity filter: 500ms window, 10x10 neighborhood; drop if no similar events in that window
ACTIVITY_WINDOW_US = 500_000  # 500 ms
ACTIVITY_RADIUS = 5  # 10x10 = radius 5 on each side
ACTIVITY_MIN_COUNT = 2  # keep event only if at least this many in window (incl. self)

# Neighborhood filter: 250ms window, 3x3 kernel; drop if sum of events in 3x3 over 250ms is not > 3
NEIGHBOR_WINDOW_US = 250_000  # 250 ms
NEIGHBOR_RADIUS = 1  # 3x3
NEIGHBOR_MIN_COUNT = 3  # keep only if count > 3 (i.e. at least 4 events in 3x3 over 250ms)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record 10s, apply activity + neighborhood filters, save RAW and show MP4.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=".",
        help="Directory for output RAW and MP4 files",
    )
    return parser.parse_args()


def record_seconds(device, output_raw_path, duration_s=RECORD_DURATION_S):
    """Record from device for duration_s seconds to a RAW file (no live preview). Returns (width, height)."""
    if device.get_i_events_stream():
        device.get_i_events_stream().log_raw_data(output_raw_path)
    mv_iterator = EventsIterator.from_device(device=device, delta_t=DELTA_T_US)
    height, width = mv_iterator.get_size()
    end_time = time.time() + duration_s
    for evs in mv_iterator:
        if time.time() >= end_time:
            break
    if device.get_i_events_stream():
        device.get_i_events_stream().stop_log_raw_data()
    return width, height


def apply_hot_pixel_mask(device, hot_pixels):
    """
    Try to mask hot pixels at the hardware level (Prophesee approach).
    - I_DigitalEventMask (Gen4.1, IMX636, GenX320): get_pixel_masks() then masks[i].set_mask(x, y, True).
    - I_RoiPixelMask (GenX320): set_pixel(x, y, False) then apply_pixels().
    Returns True if the mask was applied, False otherwise (fall back to software filtering).
    See: https://docs.prophesee.ai/stable/hw/manuals/pixel_selection/digital_event_mask.html
    """
    if not hot_pixels:
        return True
    hot_list = list(hot_pixels)[:64]  # many sensors limit to 64 mask entries
    # Prophesee I_DigitalEventMask: get_pixel_masks() returns list of I_PixelMask; each has set_mask(x, y, enabled)
    try:
        dem = device.get_i_digital_event_mask()
        if dem is not None:
            masks = dem.get_pixel_masks()
            if masks and len(masks) > 0:
                for i, (x, y) in enumerate(hot_list):
                    if i < len(masks):
                        masks[i].set_mask(int(x), int(y), True)  # True = pixel masked (disabled)
                return True
    except (AttributeError, TypeError, IndexError):
        pass
    # Alternative: some bindings expose set_pixel on the DEM directly
    try:
        dem = device.get_i_digital_event_mask()
        if dem is not None and hasattr(dem, "set_pixel"):
            for (x, y) in hot_list:
                dem.set_pixel(int(x), int(y), True)
            return True
    except (AttributeError, TypeError):
        pass
    # I_RoiPixelMask (GenX320 only)
    try:
        roi_px = device.get_i_roi_pixel_mask()
        if roi_px is not None:
            for (x, y) in hot_list:
                roi_px.set_pixel(int(x), int(y), False)  # False = pixel disabled
            roi_px.apply_pixels()
            return True
    except (AttributeError, TypeError):
        pass
    return False


def find_hot_pixels(raw_path, top_k=HOT_PIXEL_TOP_K):
    """First pass: count events per pixel, return set of (x,y) for top_k most active pixels."""
    mv_iterator = EventsIterator(input_path=raw_path, delta_t=DELTA_T_US)
    height, width = mv_iterator.get_size()
    counts = np.zeros((height, width), dtype=np.uint64)
    for evs in mv_iterator:
        if evs.size == 0:
            continue
        x, y = evs["x"], evs["y"]
        np.add.at(counts, (y, x), 1)
    flat = counts.ravel()
    # top_k linear indices
    top_linear = np.argpartition(flat, -top_k)[-top_k:]
    top_linear = top_linear[np.argsort(flat[top_linear])[::-1]]
    hot_pixels = set()
    for idx in top_linear:
        y, x = np.unravel_index(idx, counts.shape)
        hot_pixels.add((int(x), int(y)))
    return hot_pixels, width, height


def _sliding_window_filter_python(events, height, width, window_us, radius, min_count_strict):
    """Python fallback when numba not available; uses numpy for histogram slice sum."""
    n = len(events)
    if n == 0:
        return np.ones(0, dtype=bool)
    t = events["t"]
    x = events["x"]
    y = events["y"]
    keep = np.ones(n, dtype=bool)
    hist = np.zeros((height, width), dtype=np.int32)
    left = 0
    right = 0
    for i in range(n):
        t_i = t[i]
        while right < n and t[right] <= t_i + window_us:
            xr, yr = int(x[right]), int(y[right])
            if 0 <= xr < width and 0 <= yr < height:
                hist[yr, xr] += 1
            right += 1
        while left < n and t[left] < t_i - window_us:
            xl, yl = int(x[left]), int(y[left])
            if 0 <= xl < width and 0 <= yl < height:
                hist[yl, xl] -= 1
            left += 1
        xi, yi = int(x[i]), int(y[i])
        y_lo = max(0, yi - radius)
        y_hi = min(height, yi + radius + 1)
        x_lo = max(0, xi - radius)
        x_hi = min(width, xi + radius + 1)
        count = int(hist[y_lo:y_hi, x_lo:x_hi].sum())
        if count <= min_count_strict:
            keep[i] = False
    return keep


@njit(cache=True)
def _sliding_window_filter_numba(x, y, t, height, width, window_us, radius, min_count_strict):
    """JIT-compiled: sliding time window + 2D histogram. Keep iff count > min_count_strict."""
    n = x.size
    keep = np.ones(n, dtype=np.bool_)
    hist = np.zeros((height, width), dtype=np.int32)
    left = 0
    right = 0
    for i in range(n):
        t_i = t[i]
        while right < n and t[right] <= t_i + window_us:
            xr, yr = x[right], y[right]
            if 0 <= xr < width and 0 <= yr < height:
                hist[yr, xr] += 1
            right += 1
        while left < n and t[left] < t_i - window_us:
            xl, yl = x[left], y[left]
            if 0 <= xl < width and 0 <= yl < height:
                hist[yl, xl] -= 1
            left += 1
        xi, yi = x[i], y[i]
        y_lo = max(0, yi - radius)
        y_hi = min(height, yi + radius + 1)
        x_lo = max(0, xi - radius)
        x_hi = min(width, xi + radius + 1)
        count = 0
        for yy in range(y_lo, y_hi):
            for xx in range(x_lo, x_hi):
                count += hist[yy, xx]
        if count <= min_count_strict:
            keep[i] = False
    return keep


def _sliding_window_filter(events, height, width, window_us, radius, min_count_strict):
    """Use Numba JIT if available (fast), else Python fallback."""
    n = len(events)
    if n == 0:
        return np.ones(0, dtype=bool)
    if _NUMBA_AVAILABLE:
        x = np.ascontiguousarray(events["x"], dtype=np.uint16)
        y = np.ascontiguousarray(events["y"], dtype=np.uint16)
        t = np.ascontiguousarray(events["t"], dtype=np.int64)
        return _sliding_window_filter_numba(x, y, t, height, width, window_us, radius, min_count_strict)
    return _sliding_window_filter_python(events, height, width, window_us, radius, min_count_strict)


# Chunk duration for KDTree filter: process in time chunks to keep trees smaller (faster build + query).
KDTREE_CHUNK_DURATION_US = 1_000_000  # 1 second
# Use Numba sliding-window for both filters when available (single-pass O(n), much faster for large streams).
# If False, activity/neighborhood use chunked KDTree when scipy is available.
USE_NUMBA_FOR_ACTIVITY = True
USE_NUMBA_FOR_NEIGHBORHOOD = True


def _kdtree_spatiotemporal_filter(events, window_us, spatial_radius, min_count, label=""):
    """
    Filter events using a 3D KDTree in (x, y, t_scaled) with Chebyshev distance.
    Keep event i iff the number of events in the box (2*spatial_radius in x,y and window_us in t) is >= min_count.
    Processes in overlapping time chunks to reduce tree size and speed up build/query.
    """
    n = len(events)
    if n == 0:
        return np.ones(0, dtype=bool)
    x = np.ascontiguousarray(events["x"], dtype=np.float64)
    y = np.ascontiguousarray(events["y"], dtype=np.float64)
    t = np.ascontiguousarray(events["t"], dtype=np.int64)
    # Scale time so that ±window_us in t maps to ±spatial_radius in t_scaled (Chebyshev)
    t_scale = spatial_radius / float(window_us) if window_us else 1.0
    t_scaled = t * t_scale
    points = np.column_stack((x, y, t_scaled))
    radius = spatial_radius
    keep_mask = np.ones(n, dtype=bool)

    t_min, t_max = int(t[0]), int(t[-1])
    chunk_duration = max(2 * window_us, KDTREE_CHUNK_DURATION_US)
    n_chunks = max(1, (t_max - t_min + chunk_duration - 1) // chunk_duration)
    if label:
        print(f"    KDTree filter ({label}): {n_chunks} chunk(s)...")

    for chunk_start in range(t_min, t_max + 1, chunk_duration):
        chunk_end = min(chunk_start + chunk_duration, t_max + 1)
        pad_start = chunk_start - window_us
        pad_end = chunk_end + window_us
        inner_mask = (t >= chunk_start) & (t < chunk_end)
        pad_mask = (t >= pad_start) & (t < pad_end)
        inner_indices = np.where(inner_mask)[0]
        pad_indices = np.where(pad_mask)[0]
        if len(inner_indices) == 0:
            continue
        points_pad = points[pad_indices]
        tree = cKDTree(points_pad, leafsize=64)
        neighbor_counts = tree.query_ball_point(
            points[inner_indices], r=radius, p=np.inf, return_length=True
        )
        keep_mask[inner_indices] = neighbor_counts >= min_count

    return keep_mask


def activity_filter(events, height, width):
    """Keep event only if there is at least one similar event in 500ms and 10x10 neighborhood."""
    if USE_NUMBA_FOR_ACTIVITY and _NUMBA_AVAILABLE:
        return _sliding_window_filter(
            events, height, width,
            ACTIVITY_WINDOW_US, ACTIVITY_RADIUS,
            ACTIVITY_MIN_COUNT - 1,
        )
    if _SCIPY_AVAILABLE:
        return _kdtree_spatiotemporal_filter(
            events,
            window_us=ACTIVITY_WINDOW_US,
            spatial_radius=ACTIVITY_RADIUS,
            min_count=ACTIVITY_MIN_COUNT,
            label="activity",
        )
    return _sliding_window_filter(
        events, height, width,
        ACTIVITY_WINDOW_US, ACTIVITY_RADIUS,
        ACTIVITY_MIN_COUNT - 1,
    )


def neighborhood_filter(events, height, width):
    """Keep event only if sum of events in 3x3 over 250ms is > 3."""
    if USE_NUMBA_FOR_NEIGHBORHOOD and _NUMBA_AVAILABLE:
        return _sliding_window_filter(
            events, height, width,
            NEIGHBOR_WINDOW_US, NEIGHBOR_RADIUS,
            NEIGHBOR_MIN_COUNT,
        )
    if _SCIPY_AVAILABLE:
        keep = _kdtree_spatiotemporal_filter(
            events,
            window_us=NEIGHBOR_WINDOW_US,
            spatial_radius=NEIGHBOR_RADIUS,
            min_count=NEIGHBOR_MIN_COUNT + 1,
            label="neighborhood",
        )
        return keep
    return _sliding_window_filter(
        events, height, width,
        NEIGHBOR_WINDOW_US, NEIGHBOR_RADIUS,
        NEIGHBOR_MIN_COUNT,
    )


def load_events_from_raw(raw_path, hot_pixels=None, delta_t_us=DELTA_T_US):
    """Load all events from RAW, optionally excluding hot_pixels set, sorted by t. Returns (events, width, height)."""
    mv_iterator = EventsIterator(input_path=raw_path, delta_t=delta_t_us)
    height, width = mv_iterator.get_size()
    # Build 2D hot mask once for vectorized filtering (no Python loop per event)
    hot_mask = None
    if hot_pixels is not None:
        hot_mask = np.zeros((height, width), dtype=bool)
        for (hx, hy) in hot_pixels:
            if 0 <= hx < width and 0 <= hy < height:
                hot_mask[hy, hx] = True
    chunks = []
    for evs in mv_iterator:
        if evs.size == 0:
            continue
        if hot_mask is not None:
            keep = ~hot_mask[evs["y"], evs["x"]]
            evs = evs[keep]
        if evs.size:
            chunks.append(evs.copy())
    if not chunks:
        return np.array([], dtype=[("x", np.uint16), ("y", np.uint16), ("p", np.int16), ("t", np.int64)]), width, height
    events = np.concatenate(chunks)
    events = np.sort(events, order="t")
    return events, width, height


def filter_and_save_raw(
    input_raw_path, output_raw_path, hot_pixels, width, height,
    apply_activity=True, apply_neighborhood=True,
):
    """
    Load events from RAW, remove hot pixels, apply activity filter (500ms 10x10),
    then neighborhood filter (250ms 3x3), write filtered events to RAW.
    Returns (width, height, stats) where stats is a dict with n_original, n_filtered, t_min_us, t_max_us.
    """
    print("  Loading events...")
    events, w, h = load_events_from_raw(input_raw_path, hot_pixels=hot_pixels)
    width, height = w, h
    n0 = len(events)
    t_min_us = t_max_us = 0
    if n0 > 0:
        t_min_us = int(events["t"][0])
        t_max_us = int(events["t"][-1])
    if n0 == 0:
        writer = RAWEvt2EventFileWriter(
            width, height, output_raw_path,
            enable_trigger_support=False,
            metadata_map={},
            max_events_add_latency=2**63 - 1,
        )
        writer.flush()
        writer.close()
        return width, height, {"n_original": 0, "n_filtered": 0, "t_min_us": 0, "t_max_us": 0}
    if apply_activity:
        print("  Applying activity filter (500ms, 10x10)...")
        keep = activity_filter(events, height, width)
        events = events[keep]
        print(f"    After activity filter: {len(events)} events (was {n0})")
    n1 = len(events)
    if apply_neighborhood and len(events) > 0:
        print("  Applying neighborhood filter (250ms, 3x3, count > 3)...")
        keep = neighborhood_filter(events, height, width)
        events = events[keep]
        print(f"    After neighborhood filter: {len(events)} events (was {n1})")
    n_filtered = len(events)
    if n_filtered > 0:
        t_min_us = int(events["t"][0])
        t_max_us = int(events["t"][-1])
    print("  Writing filtered events to RAW...")
    writer = RAWEvt2EventFileWriter(
        width, height, output_raw_path,
        enable_trigger_support=False,
        metadata_map={},
        max_events_add_latency=2**63 - 1,
    )
    chunk_size = 100_000
    for start in range(0, len(events), chunk_size):
        end = min(start + chunk_size, len(events))
        buf = EventCDBuffer(end - start)
        buf.numpy()[:] = events[start:end]
        writer.add_cd_events(buf)
    writer.flush()
    writer.close()
    return width, height, {"n_original": n0, "n_filtered": n_filtered, "t_min_us": t_min_us, "t_max_us": t_max_us}


# HDF5 event dtype: x, y, p, t (matches Metavision / ECF EventCD)
HDF5_EVENT_DTYPE = np.dtype([("x", np.uint16), ("y", np.uint16), ("p", np.int16), ("t", np.int64)])
INDEX_INTERVAL_US = 2000  # Metavision indexes every 2000 us
# ECF codec encodes in chunks; max events per chunk is 65535
ECF_MAX_EVENTS_PER_CHUNK = 65535


def filter_and_save_hdf5(input_raw_path, hdf5_path, hot_pixels, width, height, install_root=None):
    """
    Read RAW, drop events at hot pixels, write filtered events to HDF5 from scratch.
    Uses t,x,y,p layout; tries ECF codec if the hdf5_ecf plugin is available (see
    https://github.com/prophesee-ai/hdf5_ecf), otherwise writes uncompressed.
    """
    try:
        import h5py
    except ImportError:
        raise RuntimeError("h5py is required to write HDF5. Install with: pip install h5py")

    # ECF plugin path should be set by main() so HDF5 can load filter 0x8ECF

    # First pass: read RAW, filter, collect events
    mv_iterator = EventsIterator(input_path=input_raw_path, delta_t=DELTA_T_US)
    chunks = []
    for evs in mv_iterator:
        if evs.size == 0:
            continue
        keep = np.array(
            [(x, y) not in hot_pixels for x, y in zip(evs["x"], evs["y"])],
            dtype=bool,
        )
        filtered = evs[keep]
        if filtered.size:
            chunks.append(
                np.array(
                    list(zip(filtered["x"], filtered["y"], filtered["p"], filtered["t"])),
                    dtype=HDF5_EVENT_DTYPE,
                )
            )
    if not chunks:
        events = np.array([], dtype=HDF5_EVENT_DTYPE)
    else:
        events = np.concatenate(chunks)
    # Sort by time for indexes
    if events.size:
        events = np.sort(events, order="t")

    with h5py.File(hdf5_path, "w") as f:
        f.attrs["format"] = "HDF5"
        f.attrs["geometry"] = f"{width}x{height}"

        cd = f.create_group("CD")
        n = len(events)
        # ECF encoder allows at most 65535 events per chunk
        chunk_size_ecf = min(max(n, 1), ECF_MAX_EVENTS_PER_CHUNK)
        chunk_size_plain = min(max(n, 1), 1_000_000)
        use_ecf = False
        try:
            # Try ECF filter (0x8ECF) if plugin is available; chunk size must be <= 65535
            cd.create_dataset(
                "events",
                data=events,
                dtype=HDF5_EVENT_DTYPE,
                chunks=(chunk_size_ecf,),
                compression=0x8ECF,
            )
            use_ecf = True
        except (OSError, ValueError, RuntimeError):
            cd.create_dataset("events", data=events, dtype=HDF5_EVENT_DTYPE, chunks=(chunk_size_plain,))

        # Indexes dataset: every INDEX_INTERVAL_US for Metavision-compatible seeking
        if n > 0:
            t_min, t_max = int(events["t"][0]), int(events["t"][-1])
            index_ts = np.arange(t_min, t_max + 1, INDEX_INTERVAL_US, dtype=np.int64)
            index_ids = np.searchsorted(events["t"], index_ts, side="left")
            index_ids = np.clip(index_ids, 0, n - 1)
            indexes = np.array(list(zip(index_ids, index_ts)), dtype=[("id", np.uint64), ("ts", np.int64)])
            cd.create_dataset("indexes", data=indexes)
            cd["indexes"].attrs["offset"] = 0

    return width, height


def iter_events_from_hdf5(hdf5_path, delta_t_us=DELTA_T_US):
    """
    Yield (event_array, current_time_us) chunks from our HDF5 /CD/events (t,x,y,p).
    Works with or without ECF; requires HDF5_PLUGIN_PATH for ECF files.
    """
    try:
        import h5py
    except ImportError:
        raise RuntimeError("h5py is required to read HDF5")
    with h5py.File(hdf5_path, "r") as f:
        ev_dset = f["CD"]["events"]
        events = ev_dset[:]
    if events.size == 0:
        return
    # Sort by t and yield in delta_t slices
    events = np.sort(events, order="t")
    t = events["t"]
    t0, t_end = int(t[0]), int(t[-1])
    ts = t0
    while ts < t_end:
        mask = (t >= ts) & (t < ts + delta_t_us)
        if np.any(mask):
            ev_slice = events[mask]
            # Metavision EventCD expects numpy array with x, y, p, t
            yield ev_slice, ts
        ts += delta_t_us


def events_to_mp4(event_file_path, mp4_path, width, height):
    """Render event file (RAW or HDF5 with t,x,y,p) to MP4 using periodic frame generation.
    Streams frames to disk as they are generated (when using OpenCV) to save memory and time.
    ON (polarity 1) = white, OFF (polarity 0) = blue.
    """
    frame_gen = PeriodicFrameGenerationAlgorithm(
        sensor_width=width,
        sensor_height=height,
        fps=FRAME_FPS,
        palette=ColorPalette.Dark,
    )
    # ON events = white, OFF events = blue (BGR)
    bg_bgr = (0, 0, 0)  # black background
    on_bgr = (255, 255, 255)  # white
    off_bgr = (255, 0, 0)  # blue
    try:
        frame_gen.set_colors(bg_bgr, on_bgr, off_bgr, True)
    except (TypeError, AttributeError):
        pass  # keep default palette if set_colors signature differs
    frames = []  # used only when not streaming (no OpenCV)
    frame_count = [0]

    if HAS_CV2:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(mp4_path, fourcc, FRAME_FPS, (width, height), True)

        def on_frame(ts, frame):
            writer.write(frame)
            frame_count[0] += 1
    else:
        writer = None

        def on_frame(ts, frame):
            frames.append(frame.copy())
            frame_count[0] += 1

    frame_gen.set_output_callback(on_frame)

    try:
        try:
            mv_iterator = EventsIterator(input_path=event_file_path, delta_t=DELTA_T_US)
            for evs in mv_iterator:
                frame_gen.process_events(evs)
        except Exception:
            for ev_slice, _ in iter_events_from_hdf5(event_file_path, delta_t_us=DELTA_T_US):
                frame_gen.process_events(ev_slice)

        if frame_count[0] == 0:
            raise RuntimeError("No frames generated from event file")
    finally:
        if HAS_CV2 and writer is not None:
            writer.release()

    if not HAS_CV2 and frames:
        try:
            from skvideo.io import FFmpegWriter
            w = FFmpegWriter(mp4_path, inputdict={"-r": str(FRAME_FPS)}, outputdict={"-r": str(FRAME_FPS)})
            for f in frames:
                w.writeFrame(f[:, :, ::-1])
            w.close()
        except Exception as e:
            raise RuntimeError(f"MP4 write failed: {e}") from e


def display_mp4(mp4_path):
    """Play the MP4 file in an OpenCV window (or open with system player)."""
    if not HAS_CV2:
        import subprocess
        import platform
        if platform.system() == "Darwin":
            subprocess.run(["open", mp4_path], check=False)
        elif platform.system() == "Linux":
            subprocess.run(["xdg-open", mp4_path], check=False)
        else:
            subprocess.run(["start", "", mp4_path], shell=True, check=False)
        return
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        print("Could not open", mp4_path)
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or FRAME_FPS
    delay_ms = max(1, int(1000 / fps))
    print("Playing MP4. Press any key in the window to close.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Filtered recording (hot pixels removed)", frame)
        if cv2.waitKey(delay_ms) & 0xFF != 255:
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    ts_str = time.strftime("%y%m%d_%H%M%S", time.localtime())
    raw_path = os.path.join(output_dir, f"recording_{ts_str}.raw")
    filtered_raw_path = os.path.join(output_dir, f"recording_{ts_str}_filtered.raw")
    mp4_path = os.path.join(output_dir, f"recording_{ts_str}_filtered.mp4")

    # 1) Open camera and run short calibration to identify hot pixels
    print("Opening camera...")
    device = initiate_device("")
    calib_raw_path = os.path.join(output_dir, f"calib_{ts_str}.raw")
    print(f"Calibration: recording {CALIBRATION_DURATION_S} s to identify hot pixels...")
    width, height = record_seconds(device, calib_raw_path, duration_s=CALIBRATION_DURATION_S)
    if not os.path.isfile(calib_raw_path):
        raise FileNotFoundError("Calibration recording did not produce a file.")
    print("Identifying top 10 hot pixels from calibration...")
    hot_pixels, w2, h2 = find_hot_pixels(calib_raw_path, top_k=HOT_PIXEL_TOP_K)
    width, height = w2, h2
    print("Hot pixels (x,y):", sorted(hot_pixels))
    # Mask hot pixels at hardware so they are not output during the main recording
    mask_applied = apply_hot_pixel_mask(device, hot_pixels)
    if mask_applied:
        print("Hot pixels masked at hardware; they will be excluded from the main recording.")
    else:
        print("Hardware mask not available on this device; hot pixels will be removed in software after recording.")
        print("  (Prophesee: run 'metavision_active_pixel_detection' to detect and save calibration;")
        print("   for GenX320 use default calib path; for Gen4.1/IMX636 use Digital Event Mask in camera settings.)")
    try:
        os.remove(calib_raw_path)
    except OSError:
        pass

    # 2) Record 10 seconds (with hot pixels already masked if supported)
    print(f"Recording main sequence for {RECORD_DURATION_S} seconds...")
    width, height = record_seconds(device, raw_path, duration_s=RECORD_DURATION_S)
    if not os.path.isfile(raw_path):
        raise FileNotFoundError(
            f"Recording did not produce a file at {raw_path}. "
            "Check that the camera is recording and that the path is writable."
        )
    print("Recording saved to", raw_path)

    # 3) Apply activity and neighborhood filters; only apply software hot-pixel removal if mask was not applied
    hot_pixels_for_load = None if mask_applied else hot_pixels
    original_size_bytes = os.path.getsize(raw_path)
    print("Filtering (activity + neighborhood) and writing to RAW...")
    width, height, stats = filter_and_save_raw(
        raw_path, filtered_raw_path, hot_pixels_for_load, width, height,
        apply_activity=True, apply_neighborhood=True,
    )
    filtered_size_bytes = os.path.getsize(filtered_raw_path)
    n_orig, n_filt = stats["n_original"], stats["n_filtered"]
    t_min, t_max = stats["t_min_us"], stats["t_max_us"]
    duration_s = (t_max - t_min) / 1e6 if n_filt > 0 and t_max > t_min else 0.0
    ev_per_sec = n_filt / duration_s if duration_s > 0 else 0.0

    # Filtering summary: % events and % file size removed
    if n_orig > 0:
        pct_events_removed = 100.0 * (n_orig - n_filt) / n_orig
        print(f"  Filtering removed {pct_events_removed:.1f}% of events ({n_orig - n_filt:,} of {n_orig:,}).")
    if original_size_bytes > 0:
        pct_size_removed = 100.0 * (1.0 - filtered_size_bytes / original_size_bytes)
        print(f"  Output file is {100 - pct_size_removed:.1f}% of original size ({pct_size_removed:.1f}% size reduction).")
        print(f"  Original: {original_size_bytes / (1024*1024):.2f} MB  ->  Filtered: {filtered_size_bytes / (1024*1024):.2f} MB")

    # Filtered output stats
    print("  Filtered event output:")
    print(f"    Events: {n_filt:,}")
    print(f"    Duration: {duration_s:.2f} s")
    print(f"    Event rate: {ev_per_sec:,.0f} ev/s")
    print(f"    File size: {filtered_size_bytes / (1024*1024):.2f} MB")
    print("Filtered RAW saved to", filtered_raw_path)
    os.remove(raw_path)
    print("Deleted original raw recording.")

    # 4) Render filtered events to MP4
    print("Rendering event stream to MP4...")
    events_to_mp4(filtered_raw_path, mp4_path, width, height)
    print("MP4 saved to", mp4_path)

    # 5) Display MP4
    print("Displaying MP4...")
    display_mp4(mp4_path)

    print("Done.")


if __name__ == "__main__":
    main()
