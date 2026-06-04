# Unrealbot Computer Vision Pipeline

This document outlines the computer vision techniques implemented to achieve ultra-fast, robust detection for RoboCup using a Raspberry Pi and a 200° Fisheye Camera.

## 1. YUV Chrominance Color Tracking
Instead of analyzing traditional BGR or HSV colors, the vision pipeline strips away the brightness (Luminance, `Y`) channel entirely. It operates strictly on the `U` and `V` chrominance channels. 
- **Why?** It makes the robot practically immune to shadows, specular highlights, and varying room lighting. The neon orange ball looks the exact same in the UV space whether it is under a bright fluorescent light or in a dark corner.

## 2. Zero-Copy Downsampling
When running live on the robot, we pull raw planar `YUV420` frames directly from the Picamera2 buffer.
- We slice the byte array to extract the `U` and `V` channels at exactly half-resolution without ever calling `cv2.resize()` or `cv2.cvtColor()`. 
- Processing a quarter-sized image makes the `cv2.inRange` thresholding mathematically 4x faster, allowing us to maintain high frame rates on weak Raspberry Pi hardware.

## 3. Equidistant Fisheye Distance Estimation
Standard pinhole camera math (`distance = size * focal_length / pixels`) fails catastrophically on a 200° fisheye lens because objects at the edge of the frame are heavily squashed.
- We implemented the **Equidistant Fisheye Model** (`r = f_eq * θ`).
- We convert the apparent pixel size of the ball or ArUco markers into an angular size using an adjusted focal length (`FOCAL_LENGTH_EQ = 470`), then use trigonometry to calculate the true distance.
- This results in millimeter-accurate distance readings for the ball and allies, regardless of where they sit in the ultra-wide distortion field.

## 4. Dynamic Horizon Detection
To prevent the robot from seeing "goals" or "enemies" in the audience or the ceiling, we built a dynamic field-boundary detector. While the physical horizon of a RoboCup field is defined by the **black walls**, detecting "black" is highly unreliable (shadows, dark clothing, and cables all look black). Instead, we use the green grass as a highly reliable anchor point to calculate where the walls are:
- Every frame, we find the largest continuous blob of green grass.
- We find the top-most pixel of this green field. Because the black walls sit directly on the outer edge of the grass, the top of the grass perfectly traces the bottom of the wall.
- We subtract 70 pixels (moving slightly higher into the image) to encompass the height of the walls and any tall objects (goals/robots) sitting on the edge. This new line is defined as the **Horizon**.
- A black mask is instantly drawn over the entire image above the horizon. This deletes walls, ceilings, and people before the CPU wastes any time looking for enemies or goals in the background.

## 5. Static Chassis & Lens Masking
To prevent the robot from hallucinating enemies out of its own hardware:
- We generate a static black circle mask that matches the physical barrel of the 200° fisheye lens, hiding the dark corners.
- We apply a harsh 30% crop to the very bottom of the image, permanently blinding the camera to the robot's own chassis and wiring harness.

## 6. Subtractive Enemy Detection
Instead of searching for a specific "enemy color" (since enemy robots can be built out of any material), we use a process of elimination:
- **An enemy is anything sitting on the field that is NOT grass, NOT a goal, and NOT the ball.**
- We merge the color masks of all known objects (green + blue + yellow + orange), invert the result, and apply our Dynamic Horizon. Whatever blobs remain inside the play area are classified as unknowns.

## 7. Morphological & Shape Filtering
Once we isolate the "unknown" blobs, we clean them up to ensure high-quality bounding boxes:
- **Morphological Close (`cv2.morphologyEx`)**: We use a 5x5 structural element to fill in gaps. This acts like a smudge tool, taking a fragmented robot (e.g. wheels, wires, metal plates) and stitching it into one solid, continuous detection block.
- **Aspect Ratio Filtering**: We assume robots are roughly cubic. If an object is extremely wide (`w > h * 3.5`, like a painted line or shadow) or extremely tall (`h > w * 3.5`, like an upright wire), it is discarded.
- **Area & Capping**: We ignore microscopic noise and massive glitches, and we strictly limit the output to the **2 largest enemies** to conserve processing power for Pathfinding.

## 8. Intermittent ArUco Tracking
ArUco marker detection (for identifying teammates) is extremely CPU-heavy because it requires high-resolution grayscale thresholding and corner analysis.
- We run the heavy `aruco.detectMarkers()` function only **once every 10 frames**.
- In the intervening 9 frames, we use the cached positions. At 30 FPS, a 10-frame delay is only 333ms, which is fast enough for tactical team positioning but saves immense CPU overhead compared to scanning every single frame.

## 9. Hitbox-Based Ally Protection
Because allied robots are technically "unknown objects on the field", the subtractive enemy detector will initially flag teammates as enemies.
- We take the precise coordinates of our ArUco markers, draw a safe 40-pixel radial "hitbox" around them, and cross-reference them against the enemy bounding boxes.
- Any enemy box that intersects an ArUco hitbox is instantly re-classified as a teammate and removed from the threat list.
