"""
Upgraded Banana Ripeness Detector - Enhanced Computer Vision Engine
Includes Illumination Compensation (White Balance + CLAHE),
Adaptive Background Segmentation, and Calibrated Spectral Detection.
"""

import cv2
import numpy as np
from PIL import Image
import io

class BananaRipenessDetector:
    def __init__(self):
        # Calibrated HSV Color Boundaries for real-world phone cameras
        self.green_lower = np.array([32, 30, 35], dtype=np.uint8)
        self.green_upper = np.array([88, 255, 255], dtype=np.uint8)

        self.yellow_lower = np.array([16, 35, 60], dtype=np.uint8)
        self.yellow_upper = np.array([35, 255, 255], dtype=np.uint8)

        self.brown_lower = np.array([5, 25, 20], dtype=np.uint8)
        self.brown_upper = np.array([22, 220, 105], dtype=np.uint8)
        
        self.dark_lower = np.array([0, 0, 10], dtype=np.uint8)
        self.dark_upper = np.array([180, 255, 55], dtype=np.uint8)

    def white_balance(self, img_bgr):
        """Gray-World white balancing to neutralize indoor yellow / fluorescent lighting."""
        result = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    def load_image(self, image_source):
        """Loads and pre-processes image with standardized resolution and lighting normalization."""
        if isinstance(image_source, Image.Image):
            rgb_img = np.array(image_source.convert('RGB'))
            bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        elif isinstance(image_source, (bytes, io.BytesIO)):
            if isinstance(image_source, io.BytesIO):
                image_source = image_source.getvalue()
            nparr = np.frombuffer(image_source, np.uint8)
            bgr_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        elif isinstance(image_source, str):
            bgr_img = cv2.imread(image_source)
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Unsupported image source type")
            
        h, w = bgr_img.shape[:2]
        max_dim = 800
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            bgr_img = cv2.resize(bgr_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        # Apply lighting compensation
        bgr_balanced = self.white_balance(bgr_img)
        rgb_balanced = cv2.cvtColor(bgr_balanced, cv2.COLOR_BGR2RGB)

        return bgr_balanced, rgb_balanced

    def segment_banana(self, hsv_img):
        """Extracts the banana fruit mask using multi-spectral thresholding and morphology."""
        mask_g = cv2.inRange(hsv_img, np.array([28, 20, 25]), np.array([92, 255, 255]))
        mask_y = cv2.inRange(hsv_img, np.array([14, 25, 50]), np.array([36, 255, 255]))
        mask_b = cv2.inRange(hsv_img, np.array([4, 15, 15]), np.array([25, 230, 115]))

        combined = cv2.bitwise_or(mask_g, mask_y)
        combined = cv2.bitwise_or(combined, mask_b)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=2)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        banana_mask = np.zeros(cleaned.shape, dtype=np.uint8)

        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            max_area = cv2.contourArea(contours[0])
            for cnt in contours:
                if cv2.contourArea(cnt) > max_area * 0.12 and cv2.contourArea(cnt) > 600:
                    cv2.drawContours(banana_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        else:
            banana_mask = cleaned

        return banana_mask

    def analyze(self, image_input):
        bgr_img, rgb_img = self.load_image(image_input)
        
        # Apply CLAHE to the Value channel for enhanced spot detection
        hsv_raw = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_raw)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v_enhanced = clahe.apply(v)
        hsv_img = cv2.merge([h, s, v_enhanced])

        banana_mask = self.segment_banana(hsv_img)
        total_banana_pixels = cv2.countNonZero(banana_mask)

        # Fallback if background removal was too aggressive
        if total_banana_pixels < 400:
            banana_mask = np.ones(bgr_img.shape[:2], dtype=np.uint8) * 255
            total_banana_pixels = banana_mask.size

        # Color extraction
        green_raw = cv2.inRange(hsv_img, self.green_lower, self.green_upper)
        yellow_raw = cv2.inRange(hsv_img, self.yellow_lower, self.yellow_upper)
        brown_raw1 = cv2.inRange(hsv_img, self.brown_lower, self.brown_upper)
        dark_raw = cv2.inRange(hsv_img, self.dark_lower, self.dark_upper)
        brown_raw = cv2.bitwise_or(brown_raw1, dark_raw)

        green_mask = cv2.bitwise_and(green_raw, banana_mask)
        yellow_mask = cv2.bitwise_and(yellow_raw, banana_mask)
        brown_mask = cv2.bitwise_and(brown_raw, banana_mask)

        yellow_mask = cv2.bitwise_and(yellow_mask, cv2.bitwise_not(brown_mask))
        green_mask = cv2.bitwise_and(green_mask, cv2.bitwise_not(brown_mask))

        green_px = cv2.countNonZero(green_mask)
        yellow_px = cv2.countNonZero(yellow_mask)
        brown_px = cv2.countNonZero(brown_mask)

        active_px = max(green_px + yellow_px + brown_px, 1)

        pct_green = round((green_px / active_px) * 100, 1)
        pct_yellow = round((yellow_px / active_px) * 100, 1)
        pct_brown = round((brown_px / active_px) * 100, 1)

        stage_info = self._calculate_stage(pct_green, pct_yellow, pct_brown)
        overlay_rgb = self._create_overlay(rgb_img, green_mask, yellow_mask, brown_mask, banana_mask)

        return {
            "success": True,
            "ripeness_percentage": stage_info["ripeness_pct"],
            "stage_number": stage_info["stage_number"],
            "stage_name": stage_info["stage_name"],
            "stage_badge": stage_info["badge"],
            "category": stage_info["category"],
            "color_breakdown": {
                "Green": pct_green,
                "Yellow": pct_yellow,
                "Brown": pct_brown
            },
            "estimated_shelf_life_days": stage_info["shelf_life_days"],
            "shelf_life_desc": stage_info["shelf_life_desc"],
            "taste_profile": stage_info["taste_profile"],
            "recommended_use": stage_info["recommended_use"],
            "storage_tip": stage_info["storage_tip"],
            "original_image": rgb_img,
            "overlay_image": overlay_rgb
        }

    def _calculate_stage(self, g, y, b):
        """Von Loesecke Agricultural Ripening Scale with calibrated thresholds."""
        if b >= 35:
            return {
                "stage_number": 8,
                "stage_name": "Overripe / Senescent",
                "badge": "Overripe",
                "category": "Overripe",
                "ripeness_pct": 100,
                "shelf_life_days": 1,
                "shelf_life_desc": "Consume immediately (1 day max)",
                "taste_profile": "Extremely sweet, soft texture, high floral aroma",
                "recommended_use": "Baking: Banana bread, smoothies, pancakes",
                "storage_tip": "Peel and freeze in zip bags."
            }
        elif b >= 14:
            return {
                "stage_number": 7,
                "stage_name": "Yellow with Sugar Flecks",
                "badge": "Peak Sweetness",
                "category": "Overripe" if b >= 25 else "Ripe",
                "ripeness_pct": 95,
                "shelf_life_days": 2,
                "shelf_life_desc": "1 - 2 days remaining",
                "taste_profile": "Maximum natural sugars, soft, rich sweetness",
                "recommended_use": "Direct snacking, oatmeal, smoothies",
                "storage_tip": "Store in refrigerator to slow down spots."
            }
        elif y >= 70 and g < 12:
            return {
                "stage_number": 6,
                "stage_name": "All Yellow (Ideal Fresh Ripe)",
                "badge": "Perfectly Ripe",
                "category": "Ripe",
                "ripeness_pct": 85,
                "shelf_life_days": 3,
                "shelf_life_desc": "2 - 4 days remaining",
                "taste_profile": "Balanced sweetness and firmness, classic banana taste",
                "recommended_use": "Fresh eating, lunchboxes, fruit salads",
                "storage_tip": "Store at room temperature away from apples."
            }
        elif y >= 55 and g >= 10:
            return {
                "stage_number": 5,
                "stage_name": "Yellow with Green Tips",
                "badge": "Firm Ripe",
                "category": "Ripe",
                "ripeness_pct": 72,
                "shelf_life_days": 5,
                "shelf_life_desc": "4 - 5 days remaining",
                "taste_profile": "Mild sweetness, firm texture, moderate starch",
                "recommended_use": "Fresh eating, fruit bowls, yogurt mix-ins",
                "storage_tip": "Store in an open fruit bowl at room temperature."
            }
        elif y > g:
            return {
                "stage_number": 4,
                "stage_name": "More Yellow than Green",
                "badge": "Semi-Ripe",
                "category": "Ripe" if y >= 60 else "Unripe",
                "ripeness_pct": 55,
                "shelf_life_days": 6,
                "shelf_life_desc": "5 - 7 days remaining",
                "taste_profile": "Lightly sweet, firm texture, low sugar",
                "recommended_use": "Dietary snacking, or wait 1-2 days to ripen",
                "storage_tip": "Place in paper bag to accelerate ripening."
            }
        elif g > y and y >= 20:
            return {
                "stage_number": 3,
                "stage_name": "More Green than Yellow",
                "badge": "Underripe",
                "category": "Unripe",
                "ripeness_pct": 35,
                "shelf_life_days": 8,
                "shelf_life_desc": "7 - 9 days remaining",
                "taste_profile": "Tangy, starchy, firm, low sugar",
                "recommended_use": "Cooking, green curries, gut-healthy prebiotic fiber",
                "storage_tip": "Leave on kitchen counter to ripen naturally."
            }
        elif y >= 8:
            return {
                "stage_number": 2,
                "stage_name": "Green with Trace of Yellow",
                "badge": "Early Stage",
                "category": "Unripe",
                "ripeness_pct": 20,
                "shelf_life_days": 10,
                "shelf_life_desc": "9 - 12 days remaining",
                "taste_profile": "High resistant starch, hard texture",
                "recommended_use": "Savory cooking, boiling, frying",
                "storage_tip": "Keep at room temperature (do NOT refrigerate green bananas)."
            }
        else:
            return {
                "stage_number": 1,
                "stage_name": "Solid Green (Unripe)",
                "badge": "Unripe / Starchy",
                "category": "Unripe",
                "ripeness_pct": 10,
                "shelf_life_days": 12,
                "shelf_life_desc": "10 - 14 days remaining",
                "taste_profile": "Pure resistant starch, hard texture",
                "recommended_use": "Plantain-style cooking, chips, green curries",
                "storage_tip": "Store at ambient room temperature."
            }

    def _create_overlay(self, rgb_img, green_mask, yellow_mask, brown_mask, banana_mask):
        overlay = rgb_img.copy()
        overlay[green_mask > 0] = [46, 204, 113]  # Emerald green
        overlay[yellow_mask > 0] = [241, 196, 15]  # Golden yellow
        overlay[brown_mask > 0] = [231, 76, 60]   # Crimson spots

        blended = cv2.addWeighted(overlay, 0.45, rgb_img, 0.55, 0)
        contours, _ = cv2.findContours(banana_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, (255, 255, 255), 2)
        return blended
