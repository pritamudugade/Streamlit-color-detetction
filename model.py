import streamlit as st
import cv2
import numpy as np
import torch
import time

# Load YOLOv8 model from .pt file
# Load YOLOv8 model from .pt file
model = torch.load("yolov8x.pt")['model']
model = model.to(torch.half)  # Convert to float32
model.eval()


# Load COCO class names
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")


# Function to preprocess the frame (replace this based on your YOLOv8 implementation)
def preprocess(frame):
    # Resize the frame to match the input size expected by the model
    input_size = (608, 608)  # Change this to match your model's input size
    resized_frame = cv2.resize(frame, input_size)
    
    # Normalize the pixel values
    normalized_frame = resized_frame / 255.0
    
    # Convert the frame to a PyTorch tensor
    tensor_frame = torch.from_numpy(normalized_frame.transpose((2, 0, 1))).float()  # Convert to float32

    
    # Add a batch dimension
    batched_frame = tensor_frame.unsqueeze(0)
    
    return batched_frame

# Function to postprocess the model output (replace this based on your YOLOv8 implementation)
def postprocess(output, frame):
    detections = []

    for batch in output:
        for detection in batch:
            # Assuming the structure of detection is [x, y, width, height, confidence, class_id]
            x, y, width, height, confidence, class_id = detection.tolist()

            # Calculate bounding box coordinates
            frame_height, frame_width = frame.shape[:2]  # You need to obtain frame dimensions
            x1 = int((x - width / 2) * frame_width)
            y1 = int((y - height / 2) * frame_height)
            x2 = int((x + width / 2) * frame_width)
            y2 = int((y + height / 2) * frame_height)

            # Assuming 'classes' is a list of class names
            class_name = classes[class_id]

            # Assuming 'colors' is a dictionary mapping class names to colors
            color = colors.get(class_name, (0, 255, 0))  # Default to green color

            label = f"{class_name} {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            detections.append((class_id, confidence, (x1, y1, x2, y2)))

    return detections




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


def load_and_convert_model():
    model = torch.load("yolov8x.pt")['model']
    model_fp16 = model.half()
    model_fp16.eval()
    return model_fp16

model = load_and_convert_model()

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
            .st-bw st-bg, .st-bv st-bg, .st-bq st-bg, .st-bu st-bg {
                background-color: #444;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

    st.title("Vehicle Detection and Color Detection")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi"])
    output_path = st.text_input("Enter output video path:", "output_video.mp4")

    confidence_threshold = st.slider("Set Confidence Threshold", 0.0, 1.0, 0.5, 0.01)


    if uploaded_file is not None:
        # Create a temporary file to save the uploaded video
        temp_file_path = "temp_video.mp4"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        total_frames = 0
        detected_frames = 0
        total_confidence = 0.0
        start_time = time.time()


        # Create video capture object
        video_capture = cv2.VideoCapture(temp_file_path)
        video_writer = None

        


        while True:
            ret, frame = video_capture.read()

            if not ret:
                st.warning("End of video")
                break

        
           
            input_tensor = preprocess(frame)

            # Convert input tensor to the same data type as model weights
            input_tensor = input_tensor.to(device=model.device, dtype=torch.float16)


            with torch.no_grad():
                model = model.half()
                model = model.to(input_tensor.device)

                output = model(input_tensor)
            

            for detection in output[0]:
                class_id = int(detection[5])  # Adjust the index based on the structure of the output
                confidence = detection[4]

                if confidence > confidence_threshold and classes[class_id] == 'car':
                    x_center, y_center, width, height = detection[0:4] * torch.tensor([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    x1, y1 = int(x_center - width / 2), int(y_center - height / 2)
                    x2, y2 = int(x_center + width / 2), int(y_center + height / 2)

                    roi = frame[y1:y2, x1:x2]
                    detected_color = detect_vehicle_color(roi)

                    if detected_color is not None:
                        label = f"{detected_color} Car"
                        color = (0, 255, 0)  # Green color
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        detected_frames += 1
                        total_confidence += confidence

                    
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

            total_frames += 1

        end_time = time.time()

        if video_writer is not None:
            video_writer.release()

        video_capture.release()

        # Calculate evaluation metrics
        if total_frames > 0:
            average_confidence = total_confidence / detected_frames if detected_frames > 0 else 0.0
            detection_rate = detected_frames / total_frames * 100
            processing_time = end_time - start_time

            st.write("Evaluation Metrics:")
            st.write(f"Total Frames: {total_frames}")
            st.write(f"Detected Frames: {detected_frames}")
            st.write(f"Detection Rate: {detection_rate:.2f}%")
            st.write(f"Average Confidence: {average_confidence:.2f}")
            st.write(f"Total Processing Time: {processing_time:.2f} seconds")

if __name__ == "__main__":
    model = load_and_convert_model()
    main()