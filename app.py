import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

# Load the YOLOv8 model
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

def main():
    # Add a title and description
    st.title("Brain Tumor Detection with YOLOv8")
    st.markdown("""
    Upload an MRI scan to detect and classify potential brain tumors. The model can identify: Glioma, Meningioma, No Tumor, and Pituitary tumors.
    """)
    
    # Sidebar with information
    with st.sidebar:
        st.header("Settings")
        
        # Confidence threshold slider
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Adjust the confidence threshold for detections"
        )
        
        st.header("About")
        st.info(
            "This application uses a YOLOv8 model trained on MRI images. For medical "
            "diagnosis, always consult a healthcare professional."
        )
    
    # Load the model
    model_path = "best.pt"  # Make sure this file is in the same directory as your app.py
    model = load_model(model_path)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an MRI scan", type=["jpg", "jpeg", "png"])
    
    # Process the image if uploaded
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded MRI Scan", use_column_width=True)
        
        st.write("Detecting tumors...")
        
        # Add a spinner during processing
        with st.spinner("Processing..."):
            # Run inference with confidence threshold
            results = model.predict(image, conf=confidence_threshold)[0]
            
            # Display results
            st.subheader("Detection Results")
            
            # Get the result image with bounding boxes
            result_image = results.plot()
            st.image(result_image, caption="Detection Result", use_column_width=True)
            
            # Display detection details
            if len(results.boxes) > 0:
                # Create a list to store detection data
                detections = []
                
                # Process each detection
                for box in results.boxes:
                    cls_id = int(box.cls.item())
                    conf = box.conf.item()
                    
                    # Get class name
                    cls_name = model.names[cls_id]
                    
                    # Add to detections list
                    detections.append({
                        "Tumor Type": cls_name,
                        "Confidence": f"{conf:.2%}"
                    })
                
                # Display detection table
                st.write("Detected Objects:")
                st.table(detections)
                
                # Display a summary
                st.success(f"Found {len(results.boxes)} potential tumor(s) in the image.")
            else:
                st.info("No tumors detected in this image.")

if __name__ == "__main__":
    main()
