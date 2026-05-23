import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path

from yolov8 import YOLOv8

# Page configuration
st.set_page_config(
    page_title="YOLOv8 Object Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🎯 YOLOv8 Object Detection")
st.markdown("Real-time object detection using YOLOv8 ONNX model")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Model path
model_path = st.sidebar.text_input(
    "Model Path",
    value="models/yolov8m.onnx",
    help="Path to the YOLOv8 ONNX model file"
)

# Detection parameters
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Minimum confidence score for detections"
)

iou_threshold = st.sidebar.slider(
    "IOU Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Intersection over Union threshold for NMS"
)

# Mode selection
st.sidebar.header("📋 Detection Mode")
mode = st.sidebar.radio(
    "Select Mode",
    options=["Image/Video Upload", "Webcam"],
    help="Choose between uploading files or using webcam"
)

# Initialize YOLOv8 detector
@st.cache_resource
def load_model(model_path, conf_thres, iou_thres):
    """Load and cache the YOLOv8 model"""
    try:
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found at: {model_path}")
            st.info("Please ensure the model file exists in the specified path.")
            return None
        detector = YOLOv8(model_path, conf_thres=conf_thres, iou_thres=iou_thres)
        return detector
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Load the model
yolov8_detector = load_model(model_path, conf_threshold, iou_threshold)

if yolov8_detector is None:
    st.warning("⚠️ Please configure a valid model path in the sidebar.")
    st.stop()

# Image/Video Upload Mode
if mode == "Image/Video Upload":
    st.header("📁 Upload Image or Video")
    
    uploaded_file = st.file_uploader(
        "Choose an image or video file",
        type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"],
        help="Supported formats: JPG, PNG, BMP for images; MP4, AVI, MOV, MKV for videos"
    )
    
    if uploaded_file is not None:
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        # Handle image files
        if file_extension in ['.jpg', '.jpeg', '.png', '.bmp']:
            st.subheader("🖼️ Image Detection")
            
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Create two columns for original and detected images
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Image**")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            # Perform detection
            with st.spinner("🔍 Detecting objects..."):
                boxes, scores, class_ids = yolov8_detector(image)
                detected_image = yolov8_detector.draw_detections(image)
            
            with col2:
                st.markdown("**Detected Objects**")
                st.image(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            # Display detection statistics
            st.success(f"✅ Detected {len(boxes)} objects")
            
            if len(boxes) > 0:
                st.subheader("📊 Detection Details")
                
                # Import class names from utils
                from yolov8.utils import class_names
                
                # Create detection summary
                detection_data = []
                for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
                    detection_data.append({
                        "Object #": i + 1,
                        "Class": class_names[class_id],
                        "Confidence": f"{score * 100:.2f}%",
                        "Bounding Box": f"({int(box[0])}, {int(box[1])}) - ({int(box[2])}, {int(box[3])})"
                    })
                
                st.dataframe(detection_data, use_container_width=True)
                
                # Download button for detected image
                _, img_encoded = cv2.imencode('.jpg', detected_image)
                st.download_button(
                    label="💾 Download Detected Image",
                    data=img_encoded.tobytes(),
                    file_name=f"detected_{uploaded_file.name}",
                    mime="image/jpeg"
                )
        
        # Handle video files
        elif file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
            st.subheader("🎥 Video Detection")
            
            # Save uploaded video to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_video_path = tmp_file.name
            
            # Open video
            cap = cv2.VideoCapture(tmp_video_path)
            
            if not cap.isOpened():
                st.error("❌ Error opening video file")
                os.unlink(tmp_video_path)
                st.stop()
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            st.info(f"📹 Video Info: {total_frames} frames @ {fps} FPS")
            
            # Process video option
            process_video = st.button("🚀 Start Video Detection", type="primary")
            
            if process_video:
                # Create placeholder for video frames
                frame_placeholder = st.empty()
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                frame_count = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    
                    if not ret:
                        break
                    
                    # Perform detection
                    boxes, scores, class_ids = yolov8_detector(frame)
                    detected_frame = yolov8_detector.draw_detections(frame)
                    
                    # Display frame
                    frame_placeholder.image(
                        cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_container_width=True
                    )
                    
                    # Update progress
                    frame_count += 1
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {frame_count}/{total_frames} - Detected {len(boxes)} objects")
                
                cap.release()
                os.unlink(tmp_video_path)
                
                st.success(f"✅ Video processing complete! Processed {frame_count} frames")
            else:
                cap.release()
                os.unlink(tmp_video_path)

# Webcam Mode
elif mode == "Webcam":
    st.header("📷 Real-Time Webcam Detection")
    
    st.info("💡 Click 'Start Webcam' to begin real-time object detection from your webcam")
    
    # Initialize session state for webcam control
    if 'webcam_running' not in st.session_state:
        st.session_state.webcam_running = False
    
    # Control buttons
    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("🎥 Start Webcam", type="primary", disabled=st.session_state.webcam_running)
    with col2:
        stop_button = st.button("⏹️ Stop Webcam", type="secondary", disabled=not st.session_state.webcam_running)
    
    # Handle start button
    if start_button:
        st.session_state.webcam_running = True
        st.rerun()
    
    # Handle stop button
    if stop_button:
        st.session_state.webcam_running = False
        st.rerun()
    
    # Webcam streaming
    if st.session_state.webcam_running:
        # Create placeholders for video and stats
        frame_placeholder = st.empty()
        stats_placeholder = st.empty()
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Error: Could not access webcam. Please check your camera permissions.")
            st.session_state.webcam_running = False
            st.stop()
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        st.success("✅ Webcam started successfully!")
        
        # Import class names for detection details
        from yolov8.utils import class_names
        
        # Main webcam loop
        try:
            while st.session_state.webcam_running:
                # Read frame from webcam
                ret, frame = cap.read()
                
                if not ret:
                    st.error("❌ Error: Failed to read frame from webcam")
                    break
                
                # Perform object detection
                boxes, scores, class_ids = yolov8_detector(frame)
                detected_frame = yolov8_detector.draw_detections(frame)
                
                # Convert BGR to RGB for display
                detected_frame_rgb = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
                
                # Display the frame
                frame_placeholder.image(
                    detected_frame_rgb,
                    channels="RGB",
                    use_container_width=True
                )
                
                # Display detection statistics
                if len(boxes) > 0:
                    # Count objects by class
                    class_counts = {}
                    for class_id in class_ids:
                        class_name = class_names[class_id]
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    
                    # Format statistics
                    stats_text = f"**🎯 Detected {len(boxes)} objects:** "
                    stats_text += ", ".join([f"{count} {name}" for name, count in class_counts.items()])
                    stats_placeholder.markdown(stats_text)
                else:
                    stats_placeholder.markdown("**🔍 No objects detected**")
                
                # Small delay to control frame rate and allow Streamlit to process
                # This prevents the app from becoming unresponsive
                import time
                time.sleep(0.03)  # ~30 FPS
                
        except Exception as e:
            st.error(f"❌ Error during webcam detection: {str(e)}")
        finally:
            # Clean up: release the webcam
            cap.release()
            st.session_state.webcam_running = False
    else:
        st.markdown("**Status:** Webcam is stopped. Click 'Start Webcam' to begin detection.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 About")
st.sidebar.info(
    "This application uses YOLOv8 ONNX model for real-time object detection. "
    "Upload images/videos or use your webcam to detect objects."
)

# Made with Bob
