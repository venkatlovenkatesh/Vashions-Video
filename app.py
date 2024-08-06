from flask import Flask, render_template, Response, request
import cv2
import mediapipe as mp
import numpy as np
import logging
import subprocess

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the MediaPipe face detection model
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection()

# Initialize the hand detection module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()


# Replace with the path of your default necklace image
default_necklace_image_path = 'static/Image/Necklace/necklace_1.png'
# List of ring image file paths
default_ring_image_path = 'static/Image/Ring/ring_1.png'
# Replace with the path of your default earring image
default_earring_image_path = 'static/Image/Earring/earring_1.png'
# List of bangle image file paths
default_bangle_image_path = 'static/Image/Bangle/bangle_1.png'




# Flag to check if hand is in the valid region
hand_in_frame = False

# Create a VideoCapture object to capture video from the webcam (index 0)
def get_available_cameras():
    camera_indexes = []
    for i in range(10):  # Check first 10 indexes
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            camera_indexes.append(i)
            cap.release()
    return camera_indexes

def get_video_capture():
    return cv2.VideoCapture('/dev/video0')
    
cap = get_video_capture()

def generate_frames(design, ring_image_path, necklace_image_path, earring_image_path, bangle_image_path):
    global hand_in_frame

    # Load the necklace image
    necklace_image = cv2.imread(necklace_image_path, cv2.IMREAD_UNCHANGED)
    ring_image = cv2.imread(ring_image_path, cv2.IMREAD_UNCHANGED)
    bangle_image = cv2.imread(bangle_image_path, cv2.IMREAD_UNCHANGED)
    earring_image = cv2.imread(earring_image_path, cv2.IMREAD_UNCHANGED)

    while True:
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

        # Get the width and height of the frame
        

        # Draw vertical lines to split the screen
        #cv2.line(frame, (side_width, 0), (side_width, height), (0, 255, 0), 1)
        #cv2.line(frame, (side_width + center_width, 0), (side_width + center_width, height), (0, 255, 0), 1)

            if design == "Ring Design":
                # Process the center section for hand detection and overlay
                frame_rgb = cv2.cvtColor(center_section, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                # Check if any hands were detected in the center section
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Iterate over the landmarks and draw them on the frame
                        for idx, landmark in enumerate(hand_landmarks.landmark):
                            # Get the pixel coordinates of the landmark
                            cx, cy = int(landmark.x * center_width), int(landmark.y * height)

                            # Check if hand is within the valid region
                            if cx >= 0 and cx <= center_width:
                                hand_in_frame = True

                                # Get the pixel coordinates of points 13 and 14 (ring_finger)
                                point13 = hand_landmarks.landmark[13]
                                point14 = hand_landmarks.landmark[14]
                                point13_x, point13_y = int(point13.x * center_width), int(point13.y * height)
                                point14_x, point14_y = int(point14.x * center_width), int(point14.y * height)

                                # Calculate the center coordinates and size of the bounding box for points 13 and 14
                                bbox_size = 20  # Adjust the size of the bounding box as needed
                                center_x = (point13_x + point14_x) // 2
                                center_y = (point13_y + point14_y) // 2
                                x1, y1 = center_x - bbox_size, center_y - bbox_size
                                x2, y2 = center_x + bbox_size, center_y + bbox_size

                                # Resize the image1 to fit the bounding box size
                                resized_image1 = cv2.resize(cv2.imread(ring_image_path, cv2.IMREAD_UNCHANGED), (x2 - x1, y2 - y1))
                                

                                # Define the region of interest for placing image1
                                roi_1 = center_section[y1:y2, x1:x2]

                                # Create a mask from the alpha channel of image1
                                image1_alpha = resized_image1[:, :, 3] / 255.0
                                mask_1 = np.stack([image1_alpha] * 3, axis=2)

                                # Apply the mask to image1 if the shapes match
                                if roi_1.shape == mask_1.shape:
                                    # Apply the mask to image1
                                    masked_image1 = resized_image1[:, :, :3] * mask_1

                                    # Create a mask for the region of interest of image1
                                    roi_mask_1 = 1 - mask_1

                                    # Apply the inverse mask to the region of interest
                                    roi_combined_1 = roi_1 * roi_mask_1

                                    # Combine the masked image1 and the region of interest
                                    combined_1 = cv2.add(masked_image1, roi_combined_1)

                                    # Place the combined image1 back into the center section
                                    center_section[y1:y2, x1:x2] = combined_1

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
                        if left_ear_top_left[0] >= 0 and left_ear_top_left[1] >= 0 and \
                        left_ear_bottom_right[0] <= center_section.shape[1] and left_ear_bottom_right[1] <= center_section.shape[0]:
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
                        if right_ear_top_left[0] >= 0 and right_ear_top_left[1] >= 0 and \
                        right_ear_bottom_right[0] <= center_section.shape[1] and right_ear_bottom_right[1] <= center_section.shape[0]:
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
                # Process the center section for face detection and overlay
                frame_rgb = cv2.cvtColor(center_section, cv2.COLOR_BGR2RGB)
                results = face_detection.process(frame_rgb)
                
                if results.detections:
                    for detection in results.detections:
                        # Iterate over the landmarks and draw them on the frame
                        for idx, landmark in enumerate(detection.location_data.relative_keypoints):
                            # Get the pixel coordinates of the landmark
                            cx, cy = int(landmark.x * center_width), int(landmark.y * height)

                            # Check if hand is within the valid region
                            if cx >= 0 and cx <= center_width:
                                hand_in_frame = True

                            # Extract the bounding box coordinates
                            bboxC = detection.location_data.relative_bounding_box
                            hC, wC, _ = center_section.shape
                            xminC = int(bboxC.xmin * wC)
                            yminC = int(bboxC.ymin * hC)
                            widthC = int(bboxC.width * wC)
                            heightC = int(bboxC.height * hC)
                            xmaxC = xminC + widthC
                            ymaxC = yminC + heightC

                            # Calculate the bottom bounding box coordinates
                            bottom_ymin = ymaxC + 10
                            bottom_ymax = min(ymaxC + 150, hC)

                            # Increase the width of the red bounding box
                            xminC -= 20  # Decrease the left side
                            xmaxC += 20  # Increase the right side

                            # Check if the bounding box dimensions are valid
                            if widthC > 0 and heightC > 0 and xmaxC > xminC and bottom_ymax > bottom_ymin:
                                # Resize necklace image to fit the bounding box size
                                resized_image = cv2.resize(necklace_image, (xmaxC - xminC, bottom_ymax - bottom_ymin))

                                # Calculate the start and end coordinates for the necklace image
                                start_x = xminC
                                start_y = bottom_ymin
                                end_x = start_x + (xmaxC - xminC)
                                end_y = start_y + (bottom_ymax - bottom_ymin)

                                # Create a mask from the alpha channel
                                alpha_channel = resized_image[:, :, 3]
                                mask = alpha_channel[:, :, np.newaxis] / 255.0

                                # Apply the mask to the necklace image
                                overlay = resized_image[:, :, :3] * mask

                                # Create a mask for the input image region
                                mask_inv = 1 - mask

                                # Apply the inverse mask to the input image
                                region = center_section[start_y:end_y, start_x:end_x]
                                resized_mask_inv = None
                                if region.shape[1] > 0 and region.shape[0] > 0:
                                    resized_mask_inv = cv2.resize(mask_inv, (region.shape[1], region.shape[0]))
                                    resized_mask_inv = resized_mask_inv[:, :, np.newaxis]  # Add an extra dimension to match the number of channels

                                if resized_mask_inv is not None:
                                    region_inv = region * resized_mask_inv

                                    # Combine the resized image and the input image region
                                    resized_overlay = None
                                    if region_inv.shape[1] > 0 and region_inv.shape[0] > 0:
                                        resized_overlay = cv2.resize(overlay, (region_inv.shape[1], region_inv.shape[0]))

                                    # Combine the resized overlay and region_inv
                                    region_combined = cv2.add(resized_overlay, region_inv)

                                    # Replace the neck region in the input image with the combined region
                                    center_section[start_y:end_y, start_x:end_x] = region_combined



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

            # Yield the frame as a byte string
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
        except Exception as e:
            logger.exception(f"Error in generate_frames: {str(e)}")
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        


@app.route('/')
def index():
    return render_template('img_renderr.html')


@app.route('/video_feed')
def video_feed():
    try:
        design = request.args.get('selected_accessory', 'RingDesign')
        
        if design == "RingDesign":
            ring_image_path = request.args.get('ring_image_path', default=default_ring_image_path)
            return Response(generate_frames("Ring Design", ring_image_path, default_necklace_image_path, default_earring_image_path, default_bangle_image_path), 
                            mimetype='multipart/x-mixed-replace; boundary=frame')
        
        elif design == "NecklaceDesign":
            necklace_image_path = request.args.get('necklace_image_path', default=default_necklace_image_path)
            return Response(generate_frames("Necklace Design", default_ring_image_path, necklace_image_path, default_earring_image_path, default_bangle_image_path), 
                            mimetype='multipart/x-mixed-replace; boundary=frame')
        
        elif design == "EarringDesign":
            earring_image_path = request.args.get('earring_image_path', default=default_earring_image_path)
            return Response(generate_frames("Earring Design", default_ring_image_path, default_necklace_image_path, earring_image_path, default_bangle_image_path), 
                            mimetype='multipart/x-mixed-replace; boundary=frame')
        
        elif design == "BangleDesign":
            bangle_image_path = request.args.get('bangle_image_path', default=default_bangle_image_path)
            return Response(generate_frames("Bangle Design", default_ring_image_path, default_necklace_image_path, default_earring_image_path, bangle_image_path), 
                            mimetype='multipart/x-mixed-replace; boundary=frame')

        return "Invalid design selection"
    except Exception as e:
        logger.exception(f"Error in video_feed route: {str(e)}")
        return "Error processing video feed", 500

@app.route('/health')
def health_check():
    return "Application is running", 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80, debug=True)
