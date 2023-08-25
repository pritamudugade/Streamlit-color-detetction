import streamlit as st
import cv2
import numpy as np
import time

# Load YOLOv3 model
net = cv2.dnn.readNet("yolov3-spp.weights", "yolov3-spp.cfg")

# Load COCO class names
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Function to detect vehicle color
def detect_vehicle_color(frame):
    try:
        # Convert the frame to the HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for different vehicle colors
        color_ranges = {
            'red': ([0, 100, 100], [10, 255, 255]),
            'green': ([40, 40, 40], [80, 255, 255]),
            'blue': ([100, 100, 100], [130, 255, 255]),
        }
        
        # Initialize variables to keep track of the dominant color and its count
        dominant_color = None
        max_count = 0
        
        # Iterate through the color ranges
        for color, (lower_bound, upper_bound) in color_ranges.items():
            mask = cv2.inRange(hsv_frame, np.array(lower_bound), np.array(upper_bound))
            count = np.count_nonzero(mask)
            
            if count > max_count:
                max_count = count
                dominant_color = color
        
        return dominant_color
    except Exception as e:
        print("Error in detect_vehicle_color:", e)
        return None


# Main Streamlit app
def main():
    st.set_page_config(
        page_title="Vehicle Detection and Color Detection App",
        page_icon="🚗",
        layout="wide"
    )

    st.sidebar.title("Customize Appearance")
    theme = st.sidebar.selectbox("Choose Color Theme", ["Light", "Dark"])
    font_size = st.sidebar.slider("Font Size", 10, 24, 16)
    st.write(f"<style>body {{ font-size: {font_size}px; }}</style>", unsafe_allow_html=True)

    if theme == "Dark":
        st.markdown(
            """
            <style>
            body {
                background-color: #333333;
                color: white;
            }
            .st-bw, .st-bv, .st-bq, .st-bu {
                background-color: #444;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

    st.title("Vehicle and Color Detection")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi"])
    output_path = st.text_input("Enter output video path:", "output_video.mp4")

    confidence_threshold = st.slider("Set Confidence Threshold", 0.0, 1.0, 0.5, 0.01)

    if uploaded_file is not None:
        temp_file_path = "temp_video.mp4"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        video_capture = cv2.VideoCapture(temp_file_path)
        video_writer = None

        while True:
            ret, frame = video_capture.read()

            if not ret:
                st.warning("End of video")
                break

            blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255, size=(608, 608), swapRB=True, crop=False)
            net.setInput(blob)
            layer_names = net.getUnconnectedOutLayersNames()
            detections = net.forward(layer_names)

            boxes = []
            confidences = []
            class_ids = []

            for detection in detections:
                for obj in detection:
                    scores = obj[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > confidence_threshold and classes[class_id] == 'car':
                        center_x = int(obj[0] * frame.shape[1])
                        center_y = int(obj[1] * frame.shape[0])
                        width = int(obj[2] * frame.shape[1])
                        height = int(obj[3] * frame.shape[0])

                        x = center_x - width // 2
                        y = center_y - height // 2

                        boxes.append([x, y, width, height])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Apply Non-Maximum Suppression
            indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)
            indices = indices.flatten()  # Flatten the indices array
            for i in indices:
                box = boxes[i]
                x, y, w, h = box
                label = f"{classes[class_ids[i]]} {confidences[i]:.2f}"
                color = (0, 255, 0)  # Green color
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Display the frame
            st.image(frame, channels="BGR")

            # Write the frame to the output video
            if video_writer is None:
                frame_height, frame_width, _ = frame.shape
                fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # Change to 'MP4V' codec
                video_writer = cv2.VideoWriter(output_path, fourcc, 30, (frame_width, frame_height))

            video_writer.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if video_writer is not None:
            video_writer.release()

        video_capture.release()

if __name__ == "__main__":
    main()
