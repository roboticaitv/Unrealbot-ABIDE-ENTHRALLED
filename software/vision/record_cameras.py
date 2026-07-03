"""
record_cameras.py — Record both Unrealbot cameras to separate video files for offline testing.

Usage:
    python record_cameras.py                    # Record until Ctrl+C (or 'q' in preview)
    python record_cameras.py --duration 30      # Record for 30 seconds
    python record_cameras.py --no-preview       # Record without opening a preview window
    python record_cameras.py --output-dir /path # Save to a specific folder

Output:
    cam0_YYYYMMDD_HHMMSS.avi   (front camera)
    cam1_YYYYMMDD_HHMMSS.avi   (rear camera)

These files can be played back with:
    python video_tester.py cam0_20260605_171800.avi
"""

import argparse
import os
import time
from datetime import datetime

import cv2
import numpy as np

from dual_camera import DualCamera


def yuv420_to_bgr(frame):
    """Convert a YUV420 planar frame to BGR for recording/display."""
    if frame is None:
        return None
    if len(frame.shape) == 2:
        return cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
    return frame  # Already BGR (OpenCV fallback mode)


def main():
    parser = argparse.ArgumentParser(description="Record dual cameras for offline testing.")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Recording duration in seconds. 0 = until Ctrl+C / 'q'. (default: 30.0)")
    parser.add_argument("--no-preview", action="store_true",
                        help="Disable the live preview window (useful over SSH).")
    parser.add_argument("--output-dir", type=str, default=os.path.dirname(__file__),
                        help="Directory to save recordings. (default: vision/ folder)")
    parser.add_argument("--resolution", type=str, default="832x624",
                        help="Capture resolution WxH. (default: 832x624)")
    parser.add_argument("--fps", type=float, default=24.0,
                        help="Output video FPS. (default: 24.0)")
    args = parser.parse_args()

    # Parse resolution
    w, h = map(int, args.resolution.split("x"))

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate timestamped filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path0 = os.path.join(args.output_dir, f"cam0_{timestamp}.avi")
    path1 = os.path.join(args.output_dir, f"cam1_{timestamp}.avi")

    # Initialize cameras
    print(f"Initializing cameras at {w}x{h} ...")
    cameras = DualCamera(resolution=(w, h), framerate=83, format="YUV420")
    cameras.start()
    time.sleep(1.5)  # Let auto-exposure settle

    # Set up video writers (MJPG is universally supported and fast)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer0 = cv2.VideoWriter(path0, fourcc, args.fps, (w, h))
    writer1 = cv2.VideoWriter(path1, fourcc, args.fps, (w, h))

    if not writer0.isOpened() or not writer1.isOpened():
        print("ERROR: Could not open video writers. Check codec support.")
        cameras.stop()
        return

    print(f"Recording to:")
    print(f"  CAM 0 (front): {path0}")
    print(f"  CAM 1 (rear):  {path1}")
    if args.duration > 0:
        print(f"  Duration: {args.duration:.1f}s")
    else:
        print("  Duration: until Ctrl+C or 'q'")
    print()

    frames_written = 0
    start_time = time.time()

    try:
        while True:
            # Check duration limit
            elapsed = time.time() - start_time
            if args.duration > 0 and elapsed >= args.duration:
                print(f"\nDuration limit reached ({args.duration:.1f}s).")
                break

            # Grab frames
            f0_raw, f1_raw = cameras.get_frames()
            if f0_raw is None or f1_raw is None:
                time.sleep(0.005)
                continue

            # Convert YUV420 → BGR
            bgr0 = yuv420_to_bgr(f0_raw)
            bgr1 = yuv420_to_bgr(f1_raw)

            if bgr0 is None or bgr1 is None:
                continue

            # Write frames
            writer0.write(bgr0)
            writer1.write(bgr1)
            frames_written += 1

            # Live preview
            if not args.no_preview:
                # Downscale for display to save bandwidth
                disp0 = cv2.resize(bgr0, (w // 2, h // 2))
                disp1 = cv2.resize(bgr1, (w // 2, h // 2))
                combined = cv2.hconcat([disp0, disp1])

                # Overlay recording indicator
                cv2.circle(combined, (20, 20), 8, (0, 0, 255), -1)  # Red dot
                cv2.putText(combined, f"REC  {elapsed:.1f}s  |  {frames_written} frames",
                            (35, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                cv2.imshow("Recording - Press 'q' to stop", combined)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nStopped by user.")
                    break

            # Print progress every 2 seconds
            if frames_written % 60 == 0:
                fps_actual = frames_written / max(elapsed, 0.001)
                print(f"  [{elapsed:6.1f}s] {frames_written} frames written ({fps_actual:.1f} FPS)", end="\r")

    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C.")

    # Cleanup
    writer0.release()
    writer1.release()
    cameras.stop()
    if not args.no_preview:
        cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    print(f"\nDone! Recorded {frames_written} frames in {elapsed:.1f}s")
    print(f"  {path0}  ({os.path.getsize(path0) / 1024 / 1024:.1f} MB)")
    print(f"  {path1}  ({os.path.getsize(path1) / 1024 / 1024:.1f} MB)")
    print(f"\nTo play back:")
    print(f"  python video_tester.py {path0}")
    print(f"  python video_tester.py {path1}")


if __name__ == "__main__":
    main()
