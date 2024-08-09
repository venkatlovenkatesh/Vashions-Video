from flask import Flask, render_template, Response, request
import cv2
import mediapipe as mp
import numpy as np
import logging
from collections import deque

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the MediaPipe face detection model
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Initialize the hand detection module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Pose detection
pose_detection = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Additional variables
face_bbox_buffer = deque(maxlen=10)
hand_positions = deque(maxlen=10)
upload_counter = 0
head_turn_threshold = 0.1
prev_lighting_info = None


# Replace with the path of your default necklace image
default_necklace_image_path = 'static/Image/Necklace/necklace_1.png'
# List of ring image file paths
default_ring_image_path = 'static/Image/Ring/ring_1.png'
# Replace with the path of your default earring image
default_earring_image_path = 'static/Image/Earring/earring_1.png'
# List of bangle image file paths
default_bangle_image_path = 'static/Image/Bangle/bangle_1.png'

def calculate_brightness(frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        brightness = np.mean(hsv[:, :, 2])
        return brightness

def adjust_brightness(frame, brightness):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = hsv[:, :, 2] * (brightness / 255.0)
    adjusted_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return adjusted_image

def detect_shadows(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    return thresh

def calculate_average_color(frame):
    return np.mean(frame, axis=(0, 1))

def match_to_scene_color(overlay, avg_color):
    # Ensure overlay is of the correct depth
    if overlay.dtype != np.uint8:
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Convert overlay to LAB color space for color adjustment
    lab_overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2LAB)
    
    # Extract LAB channels
    l_mean, a_mean, b_mean = cv2.split(lab_overlay)
    
    # Adjust LAB channels based on the average color
    l_mean = np.clip(l_mean + avg_color[0], 0, 255).astype(np.uint8)
    a_mean = np.clip(a_mean + avg_color[1], 0, 255).astype(np.uint8)
    b_mean = np.clip(b_mean + avg_color[2], 0, 255).astype(np.uint8)
    
    # Merge the LAB channels back to the overlay image
    adjusted_lab = cv2.merge((l_mean, a_mean, b_mean))
    
    # Convert the adjusted LAB image back to RGB
    matched_overlay = cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2RGB)
    
    return matched_overlay

def apply_shadow_mask(frame, shadow_mask):
    # Convert image to uint8 if necessary
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

    # Convert shadow_mask to uint8 if necessary
    if shadow_mask.dtype != np.uint8:
        shadow_mask = shadow_mask.astype(np.uint8)

    # Check if shadow_mask has the same size as image
    if shadow_mask.shape[:2] != frame.shape[:2]:
        shadow_mask = cv2.resize(shadow_mask, (frame.shape[1], frame.shape[0]))

    # Apply the shadow mask to the image
    masked_image = cv2.bitwise_and(frame, frame, mask=shadow_mask)
    return masked_image

def adjust_lighting(frame, brightness):
    # Example implementation (adjust brightness by scaling)
    alpha = 0.1  # Adjust this parameter based on brightness
    adjusted_image = frame * alpha
    return adjusted_image

def analyze_lighting(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    avg_l = np.mean(l)
    avg_a = np.mean(a)
    avg_b = np.mean(b)
    color_temperature = 5000 + (avg_b - avg_a) * 10
    lighting_type = 'Natural' if 4500 <= color_temperature <= 6500 else 'Artificial'
    return {
        'average_lightness': avg_l,
        'color_temperature': color_temperature,
        'lighting_type': lighting_type
    }

def analyze_light_direction(frame, shadow_mask):
    contours, _ = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    [vx, vy, x, y] = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.01, 0.01)
    light_direction = (-vx, -vy)  # The light direction is opposite to the shadow direction
    return light_direction

def merge_hdr(frame, exposure_times):
    merge_debevec = cv2.createMergeDebevec()
    hdr = merge_debevec.process(frame, times=np.array(exposure_times, dtype=np.float32))
    tonemap = cv2.createTonemapDrago(1.0, 0.7)
    ldr = tonemap.process(hdr)
    ldr_8bit = np.clip(ldr * 255, 0, 255).astype('uint8')
    return ldr_8bit

def adjust_jewelry_lighting(jewelry_image, lighting_info):
    if lighting_info['lighting_type'] == 'Natural':
        jewelry_image = adjust_for_natural_lighting(jewelry_image, lighting_info)
    else:
        jewelry_image = adjust_for_artificial_lighting(jewelry_image, lighting_info)
    return jewelry_image

def adjust_for_natural_lighting(frame, lighting_info):
    if lighting_info['lighting_type'] == 'Natural':
        # Adjust brightness and color temperature based on the scene's natural lighting
        color_temperature = lighting_info['color_temperature']
        # Example adjustment based on color temperature
        if color_temperature < 5000:
            # Add warmth to the image for lower color temperatures
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            frame[:, :, 0] = np.clip(frame[:, :, 0] + 10, 0, 255)
            frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)
        elif color_temperature > 6500:
            # Add coolness to the image for higher color temperatures
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            frame[:, :, 0] = np.clip(frame[:, :, 0] - 10, 0, 255)
            frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)
        # Example adjustment based on brightness
        brightness_factor = lighting_info['average_lightness'] / 127.5
        frame = cv2.convertScaleAbs(frame, alpha=brightness_factor, beta=0)

    return frame

def adjust_for_artificial_lighting(frame, lighting_info):
    if lighting_info['lighting_type'] == 'Artificial':
        # Example adjustment for artificial lighting
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        # Adjust color components (a and b channels) for artificial lighting
        frame[:, :, 1] = np.clip(frame[:, :, 1] + 10, 0, 255)
        frame[:, :, 2] = np.clip(frame[:, :, 2] + 5, 0, 255)
        frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)
        # Adjust brightness
        brightness_factor = lighting_info['average_lightness'] / 100.5
        frame = cv2.convertScaleAbs(frame, alpha=brightness_factor, beta=0)

    return frame

def blend_with_environment(necklace, env_color):
    alpha = 0.5
    return cv2.addWeighted(necklace, alpha, env_color, 1 - alpha, 0)

def blend_edges(frame, mask):
    blurred_image = cv2.GaussianBlur(frame, (5, 5), 0)
    blended_image = cv2.addWeighted(frame, 0.7, blurred_image, 0.3, 0)
    return blended_image

def blend_with_environment(overlay, env_color):
    alpha = 0.5
    return cv2.addWeighted(overlay, alpha, env_color, 1 - alpha, 0)

def apply_depth_of_field(frame, depth_map):
    # Example values for radius and mask
    radius = 5
    
    # Ensure kernel size is positive and odd
    kernel_size = (2 * radius) + 1
    
    # Apply Gaussian blur with valid kernel size
    blurred_image = cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigmaX=0, sigmaY=0)
    
    # Resize depth_map to match blurred_image if needed
    depth_map_resized = cv2.resize(depth_map, (blurred_image.shape[1], blurred_image.shape[0]))

    # Assuming depth_map_resized is properly resized to match blurred_image
    # Apply depth of field effect
    blurred_image *= depth_map_resized[:, :, np.newaxis]

    return blurred_image

def adjust_surface_properties(jewelry_image, lighting_info):
    if lighting_info['lighting_type'] == 'Natural':
        # Adjust reflectivity, roughness, or other properties based on natural lighting
        jewelry_image = adjust_reflectivity(jewelry_image)
    else:
        # Adjust for artificial lighting conditions
        jewelry_image = adjust_roughness(jewelry_image)
    
    return jewelry_image

def adjust_reflectivity(jewelry_image):
    # Example adjustment for reflectivity (could involve modifying color or brightness)
    # For example, increasing brightness to simulate higher reflectivity
    jewelry_image = cv2.convertScaleAbs(jewelry_image, alpha=1.2, beta=0)
    return jewelry_image

def calculate_depth_map(image):
    # Convert image to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perform Canny edge detection
    edges = cv2.Canny(gray, 100, 200)
    
    # Invert the edges to create a depth-like map
    depth_map = cv2.bitwise_not(edges)
    
    return depth_map

def adjust_roughness(jewelry_image):
    # Example adjustment for roughness (could involve blurring or sharpening)
    # For example, applying Gaussian blur to smooth surfaces
    jewelry_image = cv2.GaussianBlur(jewelry_image, (5, 5), 0)
    return jewelry_image


# Flag to check if hand is in the valid region
hand_in_frame = False

# Create a VideoCapture object to capture video from the webcam (index 0)
cap = cv2.VideoCapture(0)

def generate_frames(design, ring_image_path, necklace_image_path, earring_image_path, bangle_image_path):
    global hand_in_frame, video_active

    necklace_image = cv2.imread(necklace_image_path, cv2.IMREAD_UNCHANGED)
    ring_image = cv2.imread(ring_image_path, cv2.IMREAD_UNCHANGED)
    bangle_image = cv2.imread(bangle_image_path, cv2.IMREAD_UNCHANGED)
    earring_image = cv2.imread(earring_image_path, cv2.IMREAD_UNCHANGED)

    while video_active:
        try:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to capture frame, using blank image")
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
           
            height, width, _ = frame.shape
            center_width = int(width * 0.35)
            side_width = (width - center_width) // 2

            left_section = frame[:, :side_width]
            center_section = frame[:, side_width:side_width + center_width]
            right_section = frame[:, side_width + center_width:]

            frame_rgb = cv2.cvtColor(center_section, cv2.COLOR_BGR2RGB)

            if design == "Ring Design":
                results = hands.process(frame_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        point13 = hand_landmarks.landmark[13]
                        point14 = hand_landmarks.landmark[14]
                        point13_x, point13_y = int(point13.x * center_width), int(point13.y * height)
                        point14_x, point14_y = int(point14.x * center_width), int(point14.y * height)

                        bbox_size = 20
                        center_x = (point13_x + point14_x) // 2
                        center_y = (point13_y + point14_y) // 2
                        x1, y1 = center_x - bbox_size, center_y - bbox_size
                        x2, y2 = center_x + bbox_size, center_y + bbox_size

                        resized_ring = cv2.resize(ring_image, (x2 - x1, y2 - y1))
                        
                        alpha_channel = resized_ring[:, :, 3] / 255.0
                        overlay = resized_ring[:, :, :3] * np.stack([alpha_channel] * 3, axis=-1)
                        background = center_section[y1:y2, x1:x2] * (1 - np.stack([alpha_channel] * 3, axis=-1))
                        center_section[y1:y2, x1:x2] = overlay + background

                        hand_in_frame = True

            elif design == "Earring Design":
            # Convert frame to RGB
                rgb_frame = cv2.cvtColor(center_section, cv2.COLOR_BGR2RGB)

                # Facial landmarks
                result = face_mesh.process(rgb_frame)

                # Index numbers for the left and right ear landmarks
                left_ear_index = 177
                right_ear_index = 401

                # Flags to check if both left and right earrings are detected
                left_ear_detected = False
                right_ear_detected = False

                if result.multi_face_landmarks:
                    for facial_landmarks in result.multi_face_landmarks:
                        # Calculate bounding box coordinates for left ear
                        left_ear_landmark = facial_landmarks.landmark[left_ear_index]
                        left_ear_x = int(left_ear_landmark.x * center_section.shape[1])
                        left_ear_y = int(left_ear_landmark.y * center_section.shape[0])
                        left_ear_bbox_size = 15
                        left_ear_top_left = (left_ear_x - 10 - left_ear_bbox_size, left_ear_y - left_ear_bbox_size)
                        left_ear_bottom_right = (left_ear_x - 10 + left_ear_bbox_size, left_ear_y + left_ear_bbox_size)

                        # Check if left ear is within the central_section
                        if (left_ear_top_left[0] >= 0 and left_ear_top_left[1] >= 0 and
                            left_ear_bottom_right[0] <= center_section.shape[1] and left_ear_bottom_right[1] <= center_section.shape[0]):
                            
                            # Resize the earring image to match the size of the bounding box
                            ear_width = left_ear_bottom_right[0] - left_ear_top_left[0]
                            ear_height = left_ear_bottom_right[1] - left_ear_top_left[1]
                            resized_earring = cv2.resize(earring_image, (ear_width, ear_height))

                            # Convert earring image to a 3-channel image with alpha channel
                            resized_earring_rgb = cv2.cvtColor(resized_earring, cv2.COLOR_BGRA2BGR)

                            # Create a mask from the alpha channel
                            alpha_channel = resized_earring[:, :, 3]
                            mask = alpha_channel[:, :, np.newaxis] / 255.0

                            # Apply the mask to the resized earring image
                            overlay = resized_earring_rgb * mask

                            # Create a mask for the input image region
                            mask_inv = 1 - mask

                            # Apply the inverse mask to the input image for the left earring
                            region_left = center_section[left_ear_top_left[1]:left_ear_bottom_right[1],
                                                        left_ear_top_left[0]:left_ear_bottom_right[0]]
                            region_left_inv = region_left * mask_inv

                            # Combine the resized earring image and the input image regions
                            region_left_combined = cv2.add(overlay, region_left_inv)

                            # Replace the left ear region in the input image with the combined region for the left earring
                            center_section[left_ear_top_left[1]:left_ear_bottom_right[1],
                                        left_ear_top_left[0]:left_ear_bottom_right[0]] = region_left_combined

                            left_ear_detected = True

                        # Calculate bounding box coordinates for right ear
                        right_ear_landmark = facial_landmarks.landmark[right_ear_index]
                        right_ear_x = int(right_ear_landmark.x * center_section.shape[1])
                        right_ear_y = int(right_ear_landmark.y * center_section.shape[0])
                        right_ear_bbox_size = 15
                        right_ear_top_left = (right_ear_x + 10 - right_ear_bbox_size, right_ear_y - right_ear_bbox_size)
                        right_ear_bottom_right = (right_ear_x + 10 + right_ear_bbox_size, right_ear_y + right_ear_bbox_size)

                        # Check if right ear is within the central_section
                        if (right_ear_top_left[0] >= 0 and right_ear_top_left[1] >= 0 and
                            right_ear_bottom_right[0] <= center_section.shape[1] and right_ear_bottom_right[1] <= center_section.shape[0]):
                            
                            # Resize the earring image to match the size of the bounding box
                            ear_width = right_ear_bottom_right[0] - right_ear_top_left[0]
                            ear_height = right_ear_bottom_right[1] - right_ear_top_left[1]
                            resized_earring = cv2.resize(earring_image, (ear_width, ear_height))

                            # Convert earring image to a 3-channel image with alpha channel
                            resized_earring_rgb = cv2.cvtColor(resized_earring, cv2.COLOR_BGRA2BGR)

                            # Create a mask from the alpha channel
                            alpha_channel = resized_earring[:, :, 3]
                            mask = alpha_channel[:, :, np.newaxis] / 255.0

                            # Apply the mask to the resized earring image
                            overlay = resized_earring_rgb * mask

                            # Create a mask for the input image region
                            mask_inv = 1 - mask

                            # Apply the inverse mask to the input image for the right earring
                            region_right = center_section[right_ear_top_left[1]:right_ear_bottom_right[1],
                                                        right_ear_top_left[0]:right_ear_bottom_right[0]]
                            region_right_inv = region_right * mask_inv

                            # Combine the resized earring image and the input image regions
                            region_right_combined = cv2.add(overlay, region_right_inv)

                            # Replace the right ear region in the input image with the combined region for the right earring
                            center_section[right_ear_top_left[1]:right_ear_bottom_right[1],
                                        right_ear_top_left[0]:right_ear_bottom_right[0]] = region_right_combined

                            right_ear_detected = True

                # If both left and right earrings are detected and displayed, set hand_in_frame to True
                hand_in_frame = left_ear_detected and right_ear_detected

            elif design == "Necklace Design":
                frame_rgb = cv2.cvtColor(center_section, cv2.COLOR_BGR2RGB)
                results = face_detection.process(frame_rgb)

                if results.detections:
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        hC, wC, _ = center_section.shape
                        xminC = int(bboxC.xmin * wC)
                        yminC = int(bboxC.ymin * hC)
                        widthC = int(bboxC.width * wC)
                        heightC = int(bboxC.height * hC)
                        xmaxC = xminC + widthC
                        ymaxC = yminC + heightC
                        shoulder_ymin = ymaxC + 19
                        chest_ymax = min(ymaxC + 137, hC)
                        xminC -= -8
                        xmaxC += 2.5

                        if widthC > 0 and heightC > 0 and xmaxC > xminC and chest_ymax > shoulder_ymin:
                            face_bbox_buffer.append((xminC, yminC, xmaxC, ymaxC, shoulder_ymin, chest_ymax))
                            avg_bbox = np.mean(face_bbox_buffer, axis=0).astype(int)
                            avg_xminC, avg_yminC, avg_xmaxC, avg_ymaxC, avg_shoulder_ymin, avg_chest_ymax = avg_bbox
                            resized_necklace = cv2.resize(necklace_image, (avg_xmaxC - avg_xminC, avg_chest_ymax - avg_shoulder_ymin))

                            landmarks = pose_detection.process(frame_rgb)
                            if landmarks.pose_landmarks:
                                left_shoulder = landmarks.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
                                right_shoulder = landmarks.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]

                                dx = right_shoulder.x - left_shoulder.x
                                dy = right_shoulder.y - left_shoulder.y
                                neck_angle = np.degrees(np.arctan2(dy, dx))

                                adjusted_angle = -neck_angle + 180
                                center = (resized_necklace.shape[1] // 2, resized_necklace.shape[0] // 2)
                                matrix = cv2.getRotationMatrix2D(center, adjusted_angle, 1.0)
                                rotated_necklace = cv2.warpAffine(resized_necklace, matrix, (resized_necklace.shape[1], resized_necklace.shape[0]))

                                shadow_mask = detect_shadows(center_section)
                                if shadow_mask is not None:
                                    overlay = apply_shadow_mask(rotated_necklace, shadow_mask)
                                else:
                                    overlay = rotated_necklace

                                brightness = calculate_brightness(center_section)
                                overlay = adjust_brightness(overlay, brightness)
                                avg_color = calculate_average_color(center_section)

                                env_color = np.full(overlay.shape, avg_color, dtype=np.uint8)
                                overlay = blend_with_environment(overlay, env_color)

                                depth_map = calculate_depth_map(center_section)
                                overlay = apply_depth_of_field(overlay, depth_map)

                                lighting_info = analyze_lighting(center_section)
                                overlay = adjust_surface_properties(overlay, lighting_info)

                                if prev_lighting_info is not None:
                                    if lighting_info['lighting_type'] == 'Natural':
                                        overlay = adjust_for_natural_lighting(overlay, lighting_info)
                                    else:
                                        overlay = adjust_for_artificial_lighting(overlay, lighting_info)

                                prev_lighting_info = lighting_info

                                overlay = match_to_scene_color(overlay, avg_color)

                                mid_shoulder_x = int((left_shoulder.x + right_shoulder.x) * wC / 2)
                                mid_shoulder_y = int((left_shoulder.y + right_shoulder.y) * hC / 2)
                                necklace_start_x = max(0, (mid_shoulder_x - resized_necklace.shape[1] // 2) - 5)
                                necklace_start_y = max(0, mid_shoulder_y - 70)

                                alpha_channel = overlay[:, :, 3]
                                mask = alpha_channel[:, :, np.newaxis] / 255.0
                                overlay_rgb = overlay[:, :, :3] * mask
                                mask_inv = 1 - mask

                                necklace_end_x = min(center_section.shape[1], necklace_start_x + overlay.shape[1])
                                necklace_end_y = min(center_section.shape[0], necklace_start_y + overlay.shape[0])

                                region = center_section[necklace_start_y:necklace_end_y, necklace_start_x:necklace_end_x]
                                if region.shape[1] > 0 and region.shape[0] > 0:
                                    resized_mask_inv = cv2.resize(mask_inv[:necklace_end_y-necklace_start_y, :necklace_end_x-necklace_start_x], (region.shape[1], region.shape[0]))
                                    resized_overlay_rgb = cv2.resize(overlay_rgb[:necklace_end_y-necklace_start_y, :necklace_end_x-necklace_start_x], (region.shape[1], region.shape[0]))
                                    region_combined = (region * resized_mask_inv) + resized_overlay_rgb
                                    center_section[necklace_start_y:necklace_end_y, necklace_start_x:necklace_end_x] = region_combined

                                overlay = blend_edges(overlay, mask)
                                guidance_text = "Necklace placed successfully"
                            else:
                                guidance_text = "Pose not detected"
                else:
                    guidance_text = "Face not detected"

                if guidance_text:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    color = (0, 0, 255)
                    thickness = 2
                    text_size = cv2.getTextSize(guidance_text, font, font_scale, thickness)[0]
                    text_x = (center_section.shape[1] - text_size[0]) // 2
                    text_y = (center_section.shape[0] + text_size[1]) // 2
                    cv2.putText(center_section, guidance_text, (text_x, text_y), font, font_scale, color, thickness)

                    hand_in_frame = True            
            elif design == "Bangle Design":
                # Process the center section for hand detection and overlay
                frame_rgb = cv2.cvtColor(center_section, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                # Check if any hands were detected in the center section
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Get the coordinates of the wrist (point 0)
                        wrist = hand_landmarks.landmark[0]
                        wrist_x = int(wrist.x * center_section.shape[1])
                        wrist_y = int(wrist.y * center_section.shape[0]) + 50  # Adjust the y-coordinate by adding the offset

                        # Define the bounding box parameters
                        box_width = 120
                        box_height = 50
                        half_width = box_width // 2
                        half_height = box_height // 2
                        top_left = (wrist_x - half_width, wrist_y - half_height)
                        bottom_right = (wrist_x + half_width, wrist_y + half_height)

                        # Check if the bounding box is within the frame
                        if top_left[0] >= 0 and top_left[1] >= 0 and bottom_right[0] < center_section.shape[1] and bottom_right[1] < center_section.shape[0]:
                            # Resize the bangle image to fit the bounding box size
                            resized_bangle = cv2.resize(bangle_image, (box_width, box_height))

                            # Define the region of interest for placing the bangle image
                            roi = center_section[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

                            # Create a mask from the alpha channel of the bangle image
                            bangle_alpha = resized_bangle[:, :, 3] / 255.0
                            mask = np.stack([bangle_alpha] * 3, axis=2)

                            # Apply the mask to the bangle image
                            masked_bangle = resized_bangle[:, :, :3] * mask

                            # Create a mask for the region of interest
                            roi_mask = 1 - mask

                            # Apply the inverse mask to the region of interest
                            roi_combined = roi * roi_mask

                            # Combine the masked bangle image and the region of interest
                            combined = cv2.add(masked_bangle, roi_combined)

                            # Replace the region of interest with the combined image
                            center_section[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = combined

                hand_in_frame = True

            # Concatenate the sections back into a single frame
            frame = np.concatenate((left_section, center_section, right_section), axis=1)

            # Display the frame
            if not hand_in_frame:
                cv2.putText(frame, "You are not in the frame. Come closer to the frame.", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            # Reset the flag for the next frame
            hand_in_frame = False

            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            logger.exception(f"Error in generate_frames: {str(e)}")
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


        
def process_necklace_design(frame, necklace_image_path):
    necklace_image = cv2.imread(necklace_image_path, cv2.IMREAD_UNCHANGED)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame_rgb)
    
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            
            # Adjust necklace position
            necklace_y = y + h + 10
            necklace_width = w + 40  # Increase width to cover neck area
            necklace_height = int(necklace_width * necklace_image.shape[0] / necklace_image.shape[1])
            
            resized_necklace = cv2.resize(necklace_image, (necklace_width, necklace_height))
            
            # Ensure the necklace doesn't go out of frame
            if necklace_y + necklace_height > ih:
                necklace_height = ih - necklace_y
                resized_necklace = cv2.resize(necklace_image, (necklace_width, necklace_height))
            
            # Create a mask for the necklace
            mask = resized_necklace[:, :, 3] if resized_necklace.shape[2] == 4 else 255
            mask = cv2.resize(mask, (necklace_width, necklace_height))
            
            # Region of interest in the frame
            roi = frame[necklace_y:necklace_y+necklace_height, x-20:x+necklace_width-20]
            
            # Blend the necklace with the frame
            for c in range(0, 3):
                roi[:, :, c] = roi[:, :, c] * (1 - mask / 255.0) + \
                               resized_necklace[:, :, c] * (mask / 255.0)
            
            # Put the blended image back into the frame
            frame[necklace_y:necklace_y+necklace_height, x-20:x+necklace_width-20] = roi
    
    return frame



@app.route('/')
def index():
    return render_template('img_renderr.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    if 'frame' not in request.files:
        return "No frame part", 400
    file = request.files['frame']
    npimg = np.fromfile(file, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    design = request.form.get('design', 'NecklaceDesign')
    jewelry_path = request.form.get('jewelry_path', '')
    
    frame = process_frame_with_design(frame, design, jewelry_path)
    
    _, buffer = cv2.imencode('.jpg', frame)
    return Response(buffer.tobytes(), mimetype='image/jpeg')




def process_frame_with_design(frame, design, jewelry_path):
    if design == "RingDesign":
        frame = process_ring_design(frame, jewelry_path)
    elif design == "NecklaceDesign":
        frame = process_necklace_design(frame, jewelry_path)
    elif design == "EarringDesign":
        frame = process_earring_design(frame, jewelry_path)
    elif design == "BangleDesign":
        frame = process_bangle_design(frame, jewelry_path)
    return frame


def process_ring_design(frame, ring_image_path):
    ring_image = cv2.imread(ring_image_path, cv2.IMREAD_UNCHANGED)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            point13 = hand_landmarks.landmark[13]
            point14 = hand_landmarks.landmark[14]
            point13_x, point13_y = int(point13.x * frame.shape[1]), int(point13.y * frame.shape[0])
            point14_x, point14_y = int(point14.x * frame.shape[1]), int(point14.y * frame.shape[0])
            bbox_size = 20
            center_x = (point13_x + point14_x) // 2
            center_y = (point13_y + point14_y) // 2
            x1, y1 = center_x - bbox_size, center_y - bbox_size
            x2, y2 = center_x + bbox_size, center_y + bbox_size
            resized_ring = cv2.resize(ring_image, (x2 - x1, y2 - y1))
            frame = overlay_image(frame, resized_ring, x1, y1)
    return frame

def process_necklace_design(frame, necklace_image_path):
    original_frame = frame.copy()
    necklace_image = cv2.imread(necklace_image_path, cv2.IMREAD_UNCHANGED)
    if necklace_image is None:
        print(f"Failed to load necklace image from {necklace_image_path}")
        return frame

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame_rgb)
   
    face_bbox_buffer = []
    prev_lighting_info = None
    guidance_text = "Face not detected"
   
    try:
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                xminC = int(bboxC.xmin * iw)
                yminC = int(bboxC.ymin * ih)
                widthC = int(bboxC.width * iw)
                heightC = int(bboxC.height * ih)
                xmaxC = xminC + widthC
                ymaxC = yminC + heightC
                shoulder_ymin = ymaxC + 19
                chest_ymax = min(ymaxC + 137, ih)
                xminC -= 8
                xmaxC += 2
               
                if widthC > 0 and heightC > 0 and xmaxC > xminC and chest_ymax > shoulder_ymin:
                    face_bbox_buffer.append((xminC, yminC, xmaxC, ymaxC, shoulder_ymin, chest_ymax))
                    avg_bbox = np.mean(face_bbox_buffer, axis=0).astype(int)
                    avg_xminC, avg_yminC, avg_xmaxC, avg_ymaxC, avg_shoulder_ymin, avg_chest_ymax = avg_bbox
                    resized_necklace = cv2.resize(necklace_image, (avg_xmaxC - avg_xminC, avg_chest_ymax - avg_shoulder_ymin))
                   
                    landmarks = pose_detection.process(frame_rgb)
                    if landmarks.pose_landmarks:
                        left_shoulder = landmarks.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
                        right_shoulder = landmarks.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
                       
                        dx = right_shoulder.x - left_shoulder.x
                        dy = right_shoulder.y - left_shoulder.y
                        neck_angle = np.degrees(np.arctan2(dy, dx))
                       
                        adjusted_angle = -neck_angle + 180
                        center = (resized_necklace.shape[1] // 2, resized_necklace.shape[0] // 2)
                        matrix = cv2.getRotationMatrix2D(center, adjusted_angle, 1.0)
                        rotated_necklace = cv2.warpAffine(resized_necklace, matrix, (resized_necklace.shape[1], resized_necklace.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
                       
                        # Comment out shadow and lighting adjustments for now
                        # shadow_mask = detect_shadows(frame)
                        # if shadow_mask is not None:
                        #     overlay = apply_shadow_mask(rotated_necklace, shadow_mask)
                        # else:
                        overlay = rotated_necklace
                       
                        # brightness = calculate_brightness(frame)
                        # overlay = adjust_brightness(overlay, brightness)
                        # avg_color = calculate_average_color(frame)
                       
                        # depth_map = calculate_depth_map(frame)
                        # overlay = apply_depth_of_field(overlay, depth_map)
                       
                        # lighting_info = analyze_lighting(frame)
                        # overlay = adjust_surface_properties(overlay, lighting_info)
                       
                        # if prev_lighting_info is not None:
                        #     if lighting_info['lighting_type'] == 'Natural':
                        #         overlay = adjust_for_natural_lighting(overlay, lighting_info)
                        #     else:
                        #         overlay = adjust_for_artificial_lighting(overlay, lighting_info)
                       
                        # prev_lighting_info = lighting_info
                       
                        # overlay = match_to_scene_color(overlay, avg_color)
                       
                        mid_shoulder_x = int((left_shoulder.x + right_shoulder.x) * iw / 2)
                        mid_shoulder_y = int((left_shoulder.y + right_shoulder.y) * ih / 2)
                        necklace_start_x = max(0, (mid_shoulder_x - rotated_necklace.shape[1] // 2) - 5)
                        necklace_start_y = max(0, mid_shoulder_y - 70)
                       
                        alpha_channel = overlay[:, :, 3] if overlay.shape[2] == 4 else np.ones(overlay.shape[:2], dtype=np.uint8) * 255
                       
                        # Create a more precise mask to reduce rectangular glitches
                        _, binary_mask = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)
                        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        precise_mask = np.zeros_like(binary_mask)
                        cv2.drawContours(precise_mask, contours, -1, (255), thickness=cv2.FILLED)
                        precise_mask = cv2.GaussianBlur(precise_mask, (5, 5), 0)
                       
                        mask = precise_mask[:, :, np.newaxis] / 255.0
                        overlay_rgb = overlay[:, :, :3]
                       
                        necklace_end_x = min(frame.shape[1], necklace_start_x + overlay.shape[1])
                        necklace_end_y = min(frame.shape[0], necklace_start_y + overlay.shape[0])
                       
                        region = frame[necklace_start_y:necklace_end_y, necklace_start_x:necklace_end_x]
                        if region.shape[1] > 0 and region.shape[0] > 0:
                            resized_mask = cv2.resize(mask[:necklace_end_y-necklace_start_y, :necklace_end_x-necklace_start_x], (region.shape[1], region.shape[0]))
                            resized_overlay_rgb = cv2.resize(overlay_rgb[:necklace_end_y-necklace_start_y, :necklace_end_x-necklace_start_x], (region.shape[1], region.shape[0]))
                           
                            # Ensure resized_mask has 3 channels
                            if len(resized_mask.shape) == 2:
                                resized_mask = np.repeat(resized_mask[:, :, np.newaxis], 3, axis=2)
                           
                            # Ensure region and resized_overlay_rgb have the same number of channels
                            if len(region.shape) == 2:
                                region = cv2.cvtColor(region, cv2.COLOR_GRAY2BGR)
                            if len(resized_overlay_rgb.shape) == 2:
                                resized_overlay_rgb = cv2.cvtColor(resized_overlay_rgb, cv2.COLOR_GRAY2BGR)
                           
                            # Perform the blending operation
                            alpha = resized_mask[:,:,0]
                            blended = (resized_overlay_rgb * alpha[:,:,np.newaxis] + region * (1 - alpha[:,:,np.newaxis])).astype(np.uint8)
                           
                            # Apply subtle feathering to the edges
                            feather_amount = 3
                            feather_mask = cv2.GaussianBlur(alpha, (feather_amount*2+1, feather_amount*2+1), 0)
                            feathered_blend = (blended * feather_mask[:,:,np.newaxis] + region * (1 - feather_mask[:,:,np.newaxis])).astype(np.uint8)
                           
                            frame[necklace_start_y:necklace_end_y, necklace_start_x:necklace_end_x] = feathered_blend
                       
                        guidance_text = "Necklace placed successfully"
                    else:
                        guidance_text = "Pose not detected"
    except Exception as e:
        print(f"Error in processing: {str(e)}")
        frame = original_frame
        guidance_text = "Error in processing"

    # Add guidance text to the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 0, 255)  # Red color
    thickness = 2
    text_size = cv2.getTextSize(guidance_text, font, font_scale, thickness)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = frame.shape[0] - 30  # 30 pixels from the bottom
    cv2.putText(frame, guidance_text, (text_x, text_y), font, font_scale, color, thickness)

    return frame








def process_earring_design(frame, earring_image_path):
    earring_image = cv2.imread(earring_image_path, cv2.IMREAD_UNCHANGED)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(frame_rgb)
    if result.multi_face_landmarks:
        for facial_landmarks in result.multi_face_landmarks:
            left_ear = facial_landmarks.landmark[234]
            right_ear = facial_landmarks.landmark[454]
            left_x, left_y = int(left_ear.x * frame.shape[1]), int(left_ear.y * frame.shape[0])
            right_x, right_y = int(right_ear.x * frame.shape[1]), int(right_ear.y * frame.shape[0])
            earring_size = 30
            resized_earring = cv2.resize(earring_image, (earring_size, earring_size))
            frame = overlay_image(frame, resized_earring, left_x - earring_size // 2, left_y - earring_size // 2)
            frame = overlay_image(frame, resized_earring, right_x - earring_size // 2, right_y - earring_size // 2)
    return frame

def process_bangle_design(frame, bangle_image_path):
    bangle_image = cv2.imread(bangle_image_path, cv2.IMREAD_UNCHANGED)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            wrist = hand_landmarks.landmark[0]
            wrist_x = int(wrist.x * frame.shape[1])
            wrist_y = int(wrist.y * frame.shape[0]) + 50
            bangle_size = 120
            resized_bangle = cv2.resize(bangle_image, (bangle_size, bangle_size // 2))
            frame = overlay_image(frame, resized_bangle, wrist_x - bangle_size // 2, wrist_y - bangle_size // 4)
    return frame


def overlay_image(background, overlay, x, y):
    if overlay.shape[2] == 4:
        alpha = overlay[:, :, 3] / 255.0
        for c in range(3):
            background[y:y+overlay.shape[0], x:x+overlay.shape[1], c] = \
                (1 - alpha) * background[y:y+overlay.shape[0], x:x+overlay.shape[1], c] + \
                alpha * overlay[:, :, c]
    else:
        background[y:y+overlay.shape[0], x:x+overlay.shape[1]] = overlay
    return background

@app.route('/video_feed')
def video_feed():
    try:
        design = request.args.get('selected_accessory', 'RingDesign')
        jewelry_path = request.args.get('jewelry_path', '')

        if not jewelry_path:
            if design == "RingDesign":
                jewelry_path = default_ring_image_path
            elif design == "NecklaceDesign":
                jewelry_path = default_necklace_image_path
            elif design == "EarringDesign":
                jewelry_path = default_earring_image_path
            elif design == "BangleDesign":
                jewelry_path = default_bangle_image_path
            else:
                return "Invalid design selection"

        return Response(generate_frames(design, jewelry_path),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    except Exception as e:
        logger.exception(f"Error in video_feed route: {str(e)}")
        return "Error processing video feed", 500

@app.route('/health')
def health_check():
    return "Application is running", 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)
