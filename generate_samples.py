"""
Synthetic Demo Banana Generator
Creates 3 sample banana images (Unripe Green, Peak Ripe Yellow, and Spotted Overripe)
so you can test the system immediately without an external photo!
"""

import cv2
import numpy as np
import os

def create_banana_image(ripeness="ripe", filename="sample_ripe.jpg"):
    # Canvas size
    w, h = 600, 400
    # Background: clean light grey marble/wood table feel
    img = np.full((h, w, 3), 245, dtype=np.uint8)

    # Banana curve center points and radius
    center = (300, 200)
    axes = (220, 75)
    angle = -15

    # Base color depending on stage
    if ripeness == "green":
        # Greenish banana
        base_color = (60, 165, 85)   # BGR: Forest/fresh green
        tip_color = (40, 120, 50)
    elif ripeness == "spotted":
        # Yellow with brown spots
        base_color = (40, 210, 245)  # BGR: Golden Yellow
        tip_color = (25, 45, 80)     # BGR: Brownish tips
    else: # ripe
        base_color = (45, 220, 250)  # BGR: Bright Banana Yellow
        tip_color = (40, 150, 70)    # BGR: Slight green stem tip

    # Draw curved banana body (ellipse)
    cv2.ellipse(img, center, axes, angle, 0, 360, base_color, -1, cv2.LINE_AA)
    
    # Cut top ellipse to make a banana crescent curve
    cut_center = (300, 150)
    cut_axes = (230, 85)
    cv2.ellipse(img, cut_center, cut_axes, angle, 0, 360, (245, 245, 245), -1, cv2.LINE_AA)

    # Stems/tips
    cv2.circle(img, (110, 255), 16, tip_color, -1, cv2.LINE_AA)
    cv2.circle(img, (485, 175), 14, tip_color, -1, cv2.LINE_AA)

    # If spotted, add sugar spots
    if ripeness == "spotted":
        np.random.seed(42)
        for _ in range(65):
            sx = np.random.randint(160, 440)
            sy = np.random.randint(200, 270)
            # check if inside banana body
            if np.linalg.norm(img[sy, sx] - np.array([245, 245, 245])) > 40:
                spot_r = np.random.randint(2, 6)
                spot_color = (np.random.randint(20, 50), np.random.randint(35, 75), np.random.randint(60, 110))
                cv2.circle(img, (sx, sy), spot_r, spot_color, -1, cv2.LINE_AA)

    # Slight Gaussian blur for natural lighting
    img = cv2.GaussianBlur(img, (5, 5), 0)

    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
    cv2.imwrite(filename, img)
    return filename

if __name__ == "__main__":
    create_banana_image("green", "e:/hackathon1/samples/sample_green.jpg")
    create_banana_image("ripe", "e:/hackathon1/samples/sample_ripe.jpg")
    create_banana_image("spotted", "e:/hackathon1/samples/sample_spotted.jpg")
    print("Sample test images generated in samples/ directory!")
