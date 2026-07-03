import cv2
import os
import sys
import numpy as np
from color_tracking import ColorTracker

def extract_candidates(video_path, output_dir="dataset"):
    # Extract the name of the video file (without extension) to make a specific folder
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    base_dir = os.path.join(output_dir, video_name)
    
    # Create dataset directories
    balls_dir = os.path.join(base_dir, "balls")
    distractors_dir = os.path.join(base_dir, "distractors")
    enemies_dir = os.path.join(base_dir, "enemies")
    
    os.makedirs(balls_dir, exist_ok=True)
    os.makedirs(distractors_dir, exist_ok=True)
    os.makedirs(enemies_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return
        
    tracker = ColorTracker()
    
    frame_count = 0
    saved_count = 0
    
    print(f"Extracting candidates from {video_path}...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        # Extract from every single frame for maximum dataset density
        # if frame_count % 3 != 0:
        #     continue
            
        results = tracker.process_yuv_frame(frame)
        
        # 1. Ball Candidates
        ball_cands = results.get("ball_candidates", [])
        best_ball = results.get("ball")
        best_center = best_ball["center"] if best_ball else None
        
        for cand in ball_cands:
            cx, cy = cand["center"]
            
            # Extract 32x32 patch from the original BGR frame
            px_min, px_max = max(0, cx - 16), min(frame.shape[1], cx + 16)
            py_min, py_max = max(0, cy - 16), min(frame.shape[0], cy + 16)
            patch = frame[py_min:py_max, px_min:px_max].copy()
            
            if patch.shape[0] > 0 and patch.shape[1] > 0:
                # Pad if near the edge
                patch_32 = np.zeros((32, 32, 3), dtype=np.uint8)
                patch_32[0:patch.shape[0], 0:patch.shape[1]] = patch
                
                # Auto-label based on our current math filter!
                # If our current math filter thought this was the real ball, save it as a ball.
                # If it rejected it due to circularity/aspect ratio, save it as a distractor!
                if best_center and cx == best_center[0] and cy == best_center[1]:
                    folder = balls_dir
                else:
                    folder = distractors_dir
                    
                filename = os.path.join(folder, f"f{frame_count}_x{cx}_y{cy}.png")
                cv2.imwrite(filename, patch_32)
                saved_count += 1
                
        # 2. Enemy Candidates (For future use, just storing them in enemies/)
        enemy_cands = results.get("unknowns", [])
        for box in enemy_cands:
            ex, ey, ew, eh = box
            cx, cy = ex + ew//2, ey + eh//2
            
            px_min, px_max = max(0, cx - 16), min(frame.shape[1], cx + 16)
            py_min, py_max = max(0, cy - 16), min(frame.shape[0], cy + 16)
            patch = frame[py_min:py_max, px_min:px_max].copy()
            
            if patch.shape[0] > 0 and patch.shape[1] > 0:
                patch_32 = np.zeros((32, 32, 3), dtype=np.uint8)
                patch_32[0:patch.shape[0], 0:patch.shape[1]] = patch
                
                filename = os.path.join(enemies_dir, f"enemy_f{frame_count}_x{cx}_y{cy}.png")
                cv2.imwrite(filename, patch_32)
                
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames. Saved {saved_count} patches...")
            
    cap.release()
    print(f"\n=== DONE ===")
    print(f"Extracted {saved_count} total patches to the '{base_dir}' directory.")
    print("\nIMPORTANT NEXT STEP:")
    print(f"1. Open '{base_dir}/distractors' and look for any real balls that were accidentally rejected. Move them to the 'balls' folder.")
    print(f"2. Open '{base_dir}/balls' and look for any false positives. Move them to the 'distractors' folder.")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "video2.mp4"
    extract_candidates(path)
