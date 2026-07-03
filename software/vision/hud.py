import cv2
import numpy as np

def draw_detections(bgr_frame, detections, threats, state, action=None, cam_label="VIDEO", latency_ms=0, show_hud=True):
    """Draw bounding boxes, labels, distances, threats, and a HUD data panel."""
    hud_lines = [cam_label, f"Latency: {latency_ms:.1f}ms"]
    
    # ── Draw Detection Zone (darken excluded areas) ──
    if "detection_mask" in detections and detections["detection_mask"] is not None:
        det_mask = detections["detection_mask"]
        if det_mask.shape[:2] != bgr_frame.shape[:2]:
            det_mask = cv2.resize(det_mask, (bgr_frame.shape[1], bgr_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        excluded = (det_mask == 0)
        bgr_frame[excluded] = bgr_frame[excluded] // 3
    
    # ── Draw Field Contour (Semi-Transparent Green) ──
    if "field_contour" in detections and detections["field_contour"] is not None:
        overlay = bgr_frame.copy()
        cv2.drawContours(overlay, [detections["field_contour"]], -1, (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.25, bgr_frame, 0.75, 0, bgr_frame)
        
    # ── Draw Horizon ──
    if "horizon_poly" in detections and detections["horizon_poly"] is not None:
        pts = np.array(detections["horizon_poly"], np.int32).reshape((-1, 1, 2))
        cv2.polylines(bgr_frame, [pts], isClosed=False, color=(255, 0, 255), thickness=2)
        cv2.putText(bgr_frame, "CURVED HORIZON", (10, pts[0][0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    elif "horizon_y" in detections:
        hy = detections["horizon_y"]
        cv2.line(bgr_frame, (0, hy), (bgr_frame.shape[1], hy), (255, 0, 255), 2)
        cv2.putText(bgr_frame, "HORIZON", (10, hy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    
    # ── Ball ──
    if detections.get("ball"):
        x, y, w, h = detections["ball"]["bbox"]
        cv2.rectangle(bgr_frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
        
        # Convert the Neural Network normalized value back into physical meters
        dist_m = state.get('ball_distance_norm', 0)
        hud_lines.append(f"Ball Dist: {dist_m:.2f}m")
        hud_lines.append(f"Ball Ang:  {state.get('ball_angle_deg', 0):.1f} deg")
        hud_lines.append(f"Ball Spd:  {state.get('ball_speed_norm', 0):.2f} m/s")
        hud_lines.append(f"Shot Opp:  {state.get('shot_opportunity_ego', 0):.2f}")
    else:
        hud_lines.append("Ball: ---")
        
    # ── Goals ──
    from vision_config import config
    enemy_color = config["physics"].get("team_color", "blue")
    
    if detections.get("blue_goal"):
        x, y, w, h = detections["blue_goal"]["bbox"]
        cv2.rectangle(bgr_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        is_enemy = (enemy_color == "blue")
        label = "ENEMY GOAL" if is_enemy else "ALLIED GOAL"
        label_color = (0, 0, 255) if is_enemy else (0, 255, 0)
        cv2.putText(bgr_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(bgr_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 1, cv2.LINE_AA)
        
        dist_m = state.get("blue_goal_distance_m", 0)
        ang_deg = state.get("blue_goal_angle_deg", 0)
        hud_lines.append(f"Blue Goal ({label[:4]}): {dist_m:.1f}m @ {ang_deg:.1f} deg")
    else:
        hud_lines.append("Blue Goal: ---")
        
    if detections.get("yellow_goal"):
        x, y, w, h = detections["yellow_goal"]["bbox"]
        cv2.rectangle(bgr_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        
        is_enemy = (enemy_color == "yellow")
        label = "ENEMY GOAL" if is_enemy else "ALLIED GOAL"
        label_color = (0, 0, 255) if is_enemy else (0, 255, 0)
        cv2.putText(bgr_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(bgr_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 1, cv2.LINE_AA)
        
        dist_m = state.get("yellow_goal_distance_m", 0)
        ang_deg = state.get("yellow_goal_angle_deg", 0)
        hud_lines.append(f"Yell Goal ({label[:4]}): {dist_m:.1f}m @ {ang_deg:.1f} deg")
    else:
        hud_lines.append("Yell Goal: ---")
        
    # ── White Lines ──
    if "white_lines" in detections and detections["white_lines"]:
        for line in detections["white_lines"]:
            # Increased contour drawing thickness from 2 to 3 for higher visibility
            cv2.drawContours(bgr_frame, [line["contour"]], -1, (255, 255, 0), 3) # Cyan
            cx, cy = line["center"]
            # Draw black contrast shadow behind "LINE" text label
            cv2.putText(bgr_frame, "LINE", (cx - 15, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(bgr_frame, "LINE", (cx - 15, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
        hud_lines.append(f"Lines Det: {len(detections['white_lines'])}")
    else:
        hud_lines.append("Lines Det: 0")


    # ── Enemies (Threats) ──
    if len(threats) > 0:
        enemy_m = state.get('enemy1_distance_norm', 0)
        hud_lines.append(f"E1 Dist: {enemy_m:.2f}m")
        hud_lines.append(f"E1 Spd: {state.get('enemy1_velocity_norm', 0):.2f}")
        for i, (x, y, w, h) in enumerate(threats):
            cv2.rectangle(bgr_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(bgr_frame, f"ENEMY", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
    # ── AI Actions ──
    if action:
        hud_lines.append(f"CMD Vx: {action.get('vx', 0):.2f}")
        hud_lines.append(f"CMD Vy: {action.get('vy', 0):.2f}")
        hud_lines.append(f"CMD W:  {action.get('omega', 0):.2f}")
        hud_lines.append(f"CMD Kick: {action.get('kick', 0):.2f}")
        
    # ── Candidates (Stage 2 Network Crops) ──
    y_offset = 10
    x_offset = bgr_frame.shape[1] - 40
    
    # Draw ball candidates
    ball_cands = detections.get("ball_candidates", [])
    if ball_cands:
        for cand in ball_cands:
            cx, cy = cand["center"]
            px_min, px_max = max(0, cx - 16), min(bgr_frame.shape[1], cx + 16)
            py_min, py_max = max(0, cy - 16), min(bgr_frame.shape[0], cy + 16)
            patch = bgr_frame[py_min:py_max, px_min:px_max]
            
            if patch.shape[0] > 0 and patch.shape[1] > 0:
                patch_32 = np.zeros((32, 32, 3), dtype=np.uint8)
                patch_32[0:patch.shape[0], 0:patch.shape[1]] = patch
                cv2.rectangle(patch_32, (0,0), (31,31), (0, 165, 255), 1) # Orange border
                
                if y_offset + 32 < bgr_frame.shape[0]:
                    bgr_frame[y_offset:y_offset+32, x_offset:x_offset+32] = patch_32
                    y_offset += 40
                    
    # Draw enemy candidates
    enemy_cands = detections.get("enemy_candidates", [])
    if enemy_cands:
        for cand in enemy_cands:
            ex, ey, ew, eh = cand["bbox"]
            cx, cy = cand["center"]
            px_min, px_max = max(0, cx - 16), min(bgr_frame.shape[1], cx + 16)
            py_min, py_max = max(0, cy - 16), min(bgr_frame.shape[0], cy + 16)
            patch = bgr_frame[py_min:py_max, px_min:px_max]
            
            if patch.shape[0] > 0 and patch.shape[1] > 0:
                patch_32 = np.zeros((32, 32, 3), dtype=np.uint8)
                patch_32[0:patch.shape[0], 0:patch.shape[1]] = patch
                cv2.rectangle(patch_32, (0,0), (31,31), (0, 0, 255), 1) # Red border
                
                if y_offset + 32 < bgr_frame.shape[0]:
                    bgr_frame[y_offset:y_offset+32, x_offset:x_offset+32] = patch_32
                    y_offset += 40
    
    # ── Draw HUD panel ──
    if show_hud:
        line_h = 36
        
        for i, line in enumerate(hud_lines):
            color = (0, 255, 0) if i == 0 else ((0, 255, 255) if i == 1 else (255, 255, 255))
            # Draw black outline/shadow for maximum readability without a background panel
            cv2.putText(bgr_frame, line, (10, 30 + i * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(bgr_frame, line, (10, 30 + i * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
    
    return bgr_frame
