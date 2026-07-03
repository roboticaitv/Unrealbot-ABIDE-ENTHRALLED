"""
Comprehensive color calibration tool.
Pulls a frame from the Pi's MJPEG stream and samples UV values
at key locations across both cameras.

Usage: python debug_colors.py
       Click on the image to sample UV values at any point.
       Press 'q' to quit.
"""
import cv2
import urllib.request
import numpy as np

STREAM_URL = 'http://192.168.137.80:8080/stream'

# Current bounds for reference
BOUNDS = {
    'orange (ball)':  {'u_min': 50,  'u_max': 125, 'v_min': 130, 'v_max': 255},
    'blue (goal)':    {'u_min': 135, 'u_max': 255, 'v_min': 60,  'v_max': 130},
    'yellow (goal)':  {'u_min': 0,   'u_max': 110, 'v_min': 100, 'v_max': 145},
    'green (field)':  {'u_min': 100, 'u_max': 135, 'v_min': 10,  'v_max': 115},
    'black (walls)':  {'u_min': 115, 'u_max': 141, 'v_min': 115, 'v_max': 141},
}

def get_stream_frame(url):
    stream = urllib.request.urlopen(url)
    bytes_data = b''
    while True:
        bytes_data += stream.read(4096)
        a = bytes_data.find(b'\xff\xd8')
        b = bytes_data.find(b'\xff\xd9')
        if a != -1 and b != -1:
            jpg = bytes_data[a:b+2]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            stream.close()
            return frame

def sample_region(yuv, x, y, size=15):
    """Sample a region and return U/V stats."""
    h, w = yuv.shape[:2]
    x1 = max(0, x - size)
    x2 = min(w, x + size)
    y1 = max(0, y - size)
    y2 = min(h, y + size)
    patch = yuv[y1:y2, x1:x2]
    u = patch[:, :, 1]
    v = patch[:, :, 2]
    return {
        'u_mean': np.mean(u), 'u_min': int(np.min(u)), 'u_max': int(np.max(u)),
        'v_mean': np.mean(v), 'v_min': int(np.min(v)), 'v_max': int(np.max(v)),
    }

def classify_color(u_mean, v_mean):
    """Check which bounds this color falls into."""
    matches = []
    for name, b in BOUNDS.items():
        if b['u_min'] <= u_mean <= b['u_max'] and b['v_min'] <= v_mean <= b['v_max']:
            matches.append(name)
    return matches if matches else ['UNCLASSIFIED']

# ── Pull frame ──────────────────────────────────────────────
print("Fetching frame from live stream...")
frame = get_stream_frame(STREAM_URL)
h, full_w = frame.shape[:2]
cam_w = full_w // 2

cam0_bgr = frame[:, :cam_w]
cam1_bgr = frame[:, cam_w:]

cam0_yuv = cv2.cvtColor(cam0_bgr, cv2.COLOR_BGR2YUV)
cam1_yuv = cv2.cvtColor(cam1_bgr, cv2.COLOR_BGR2YUV)

# ── Auto-sample key regions ────────────────────────────────
regions = {
    'CAM0': {
        'field_center':    (cam_w // 2, int(h * 0.75)),
        'field_left':      (int(cam_w * 0.25), int(h * 0.70)),
        'field_right':     (int(cam_w * 0.75), int(h * 0.70)),
        'wall_center':     (cam_w // 2, int(h * 0.30)),
        'wall_left':       (int(cam_w * 0.20), int(h * 0.35)),
        'top_center':      (cam_w // 2, int(h * 0.15)),
        'goal_area':       (cam_w // 2, int(h * 0.45)),
        'ball_area':       (int(cam_w * 0.45), int(h * 0.60)),
    },
    'CAM1': {
        'field_center':    (cam_w // 2, int(h * 0.75)),
        'field_left':      (int(cam_w * 0.25), int(h * 0.70)),
        'field_right':     (int(cam_w * 0.75), int(h * 0.70)),
        'wall_center':     (cam_w // 2, int(h * 0.30)),
        'wall_left':       (int(cam_w * 0.20), int(h * 0.35)),
        'top_center':      (cam_w // 2, int(h * 0.15)),
        'goal_area':       (cam_w // 2, int(h * 0.55)),
        'ball_area':       (int(cam_w * 0.45), int(h * 0.65)),
    }
}

print("\n" + "=" * 70)
print("COLOR CALIBRATION REPORT")
print("=" * 70)

for cam_name, yuv in [('CAM0', cam0_yuv), ('CAM1', cam1_yuv)]:
    print(f"\n{'-' * 35} {cam_name} {'-' * 35}")
    for region_name, (rx, ry) in regions[cam_name].items():
        s = sample_region(yuv, rx, ry)
        classification = classify_color(s['u_mean'], s['v_mean'])
        in_bounds = ', '.join(classification)
        print(f"  {region_name:18s} @ ({rx:3d},{ry:3d})  "
              f"U=[{s['u_min']:3d}-{s['u_max']:3d}] avg={s['u_mean']:5.1f}  "
              f"V=[{s['v_min']:3d}-{s['v_max']:3d}] avg={s['v_mean']:5.1f}  "
              f"-> {in_bounds}")

print("\n" + "=" * 70)
print("CURRENT BOUNDS REFERENCE")
print("=" * 70)
for name, b in BOUNDS.items():
    print(f"  {name:18s}  U=[{b['u_min']:3d}-{b['u_max']:3d}]  V=[{b['v_min']:3d}-{b['v_max']:3d}]")

# ── Interactive click-to-sample ─────────────────────────────
print("\n" + "=" * 70)
print("INTERACTIVE MODE: Click anywhere on the image to sample UV values.")
print("Press 'q' to quit.")
print("=" * 70)

display = frame.copy()

# Draw sample points on the display
for cam_name, cam_regions in regions.items():
    offset = 0 if cam_name == 'CAM0' else cam_w
    for region_name, (rx, ry) in cam_regions.items():
        cv2.circle(display, (rx + offset, ry), 5, (0, 255, 255), -1)
        cv2.putText(display, region_name, (rx + offset + 8, ry - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cam = 'CAM0' if x < cam_w else 'CAM1'
        yuv = cam0_yuv if x < cam_w else cam1_yuv
        local_x = x if x < cam_w else x - cam_w
        s = sample_region(yuv, local_x, y)
        classification = classify_color(s['u_mean'], s['v_mean'])
        print(f"\n  CLICK {cam} @ ({local_x},{y})  "
              f"U=[{s['u_min']}-{s['u_max']}] avg={s['u_mean']:.1f}  "
              f"V=[{s['v_min']}-{s['v_max']}] avg={s['v_mean']:.1f}  "
              f"-> {', '.join(classification)}")
        # Draw on display
        cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Color Calibration', display)

cv2.namedWindow('Color Calibration', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Color Calibration', on_mouse)
cv2.imshow('Color Calibration', display)

while True:
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
