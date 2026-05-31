def boxes_intersect(box1, box2, padding=0):
    """
    Checks if two bounding boxes intersect mathematically (Axis-Aligned Bounding Box).
    Takes a fraction of a microsecond compared to OpenCV pixel dilations.
    
    box format: (x, y, width, height)
    padding: Expands box1 by this many pixels in all directions (simulates dilation).
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Expand box1 by the padding (hitbox dilation)
    b1_min_x = x1 - padding
    b1_max_x = x1 + w1 + padding
    b1_min_y = y1 - padding
    b1_max_y = y1 + h1 + padding

    b2_min_x = x2
    b2_max_x = x2 + w2
    b2_min_y = y2
    b2_max_y = y2 + h2

    # Standard AABB intersection test
    if (b1_min_x < b2_max_x and b1_max_x > b2_min_x and
        b1_min_y < b2_max_y and b1_max_y > b2_min_y):
        return True
    
    return False

def filter_threats(unknown_contours, ally_boxes, safe_radius=30):
    """
    Filters out any unknown object that is just a piece of our own robot.
    Returns a list of true enemy bounding boxes.
    """
    threats = []
    for cnt_box in unknown_contours:
        is_ally = False
        for ally in ally_boxes:
            if boxes_intersect(ally, cnt_box, padding=safe_radius):
                is_ally = True
                break
        
        if not is_ally:
            threats.append(cnt_box)
            
    return threats

if __name__ == "__main__":
    # Test cases
    ally = (100, 100, 50, 50)
    threat = (160, 100, 20, 20) # 10 pixels away
    
    # Without padding, they don't intersect
    print("Intersect without padding:", boxes_intersect(ally, threat)) # False
    
    # With 15 padding, they do intersect (160 < 100+50+15 = 165)
    print("Intersect with padding:", boxes_intersect(ally, threat, padding=15)) # True
