# app.py - Main Streamlit Application
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import os
from datetime import datetime
import json
import zipfile
import tempfile
import shutil
from PIL import Image
import io

# Page config
st.set_page_config(
    page_title="Motorcycle Privacy Blur Tool",
    page_icon="🏍️",
    layout="wide"
)

# Initialize session state
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'model' not in st.session_state:
    st.session_state.model = None

class PrivacyBlurProcessor:
    def __init__(self, blur_strength=99, blur_method='head_region'):
        """Initialize the processor"""
        if st.session_state.model is None:
            with st.spinner("Loading AI model... (first time only)"):
                st.session_state.model = YOLO('yolov8n.pt')
        
        self.model = st.session_state.model
        self.blur_strength = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
        self.blur_method = blur_method
    
    def get_blur_region(self, person_bbox, method='head_region'):
        """Calculate region to blur based on person detection"""
        x1, y1, x2, y2 = person_bbox
        person_height = y2 - y1
        
        if method == 'head_region':
            blur_height = int(person_height * 0.3)
            return (x1, y1, x2, y1 + blur_height)
        elif method == 'upper_body':
            blur_height = int(person_height * 0.5)
            return (x1, y1, x2, y1 + blur_height)
        elif method == 'full_person':
            return (x1, y1, x2, y2)
        
        return person_bbox
    
    def process_image(self, image):
        """Process a single image (numpy array)"""
        # Detect persons
        results = self.model(image, classes=[0])  # Class 0 is 'person'
        
        persons_found = 0
        result_img = image.copy()
        detections = []
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf[0].cpu().numpy()
                    
                    # Get blur region
                    blur_region = self.get_blur_region((x1, y1, x2, y2), self.blur_method)
                    bx1, by1, bx2, by2 = blur_region
                    
                    # Extract and blur region
                    region = result_img[by1:by2, bx1:bx2]
                    if region.size > 0:
                        blurred = cv2.GaussianBlur(region, (self.blur_strength, self.blur_strength), 0)
                        blurred = cv2.GaussianBlur(blurred, (self.blur_strength, self.blur_strength), 0)
                        result_img[by1:by2, bx1:bx2] = blurred
                        persons_found += 1
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'blur_region': (bx1, by1, bx2, by2),
                            'confidence': float(conf)
                        })
        
        return result_img, persons_found, detections

def main():
    st.title("🏍️ Motorcycle Privacy Blur Tool")
    st.markdown("Automatically blur faces/heads of motorcycle riders for privacy compliance")
    
    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        blur_method = st.selectbox(
            "Blur Region",
            options=['head_region', 'upper_body', 'full_person'],
            format_func=lambda x: {
                'head_region': 'Head/Helmet Only (30%)',
                'upper_body': 'Upper Body (50%)',
                'full_person': 'Full Person (100%)'
            }[x]
        )
        
        blur_strength = st.slider(
            "Blur Intensity",
            min_value=21,
            max_value=199,
            value=99,
            step=20,
            help="Higher values = stronger blur"
        )
        
        # Ensure odd number
        if blur_strength % 2 == 0:
            blur_strength += 1
        
        st.markdown("---")
        
        process_type = st.radio(
            "Process",
            options=['images', 'videos', 'both'],
            format_func=lambda x: {
                'images': '🖼️ Images Only',
                'videos': '🎬 Videos Only',
                'both': '📁 Both'
            }[x]
        )
        
        st.markdown("---")
        st.markdown("### 📊 Session Stats")
        st.metric("Files Processed", len(st.session_state.processed_files))
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "🔍 Preview Results", "📚 Instructions"])
    
    with tab1:
        st.header("Upload Files")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['png', 'jpg', 'jpeg', 'bmp', 'mp4', 'avi', 'mov'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} files uploaded")
            
            # Process button
            if st.button("🚀 Start Processing", type="primary"):
                processor = PrivacyBlurProcessor(blur_strength=blur_strength, blur_method=blur_method)
                
                # Create progress containers
                progress_bar = st.progress(0)
                status_text = st.empty()
                results_container = st.container()
                
                processed_images = []
                total_persons = 0
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    progress = (idx + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    # Check file type
                    is_image = uploaded_file.type.startswith('image')
                    is_video = uploaded_file.type.startswith('video')
                    
                    # Skip based on process_type setting
                    if process_type == 'images' and not is_image:
                        continue
                    elif process_type == 'videos' and not is_video:
                        continue
                    
                    if is_image:
                        # Process image
                        image = Image.open(uploaded_file)
                        image_np = np.array(image)
                        
                        # Convert RGB to BGR for OpenCV
                        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                        
                        # Process
                        result_img, persons_count, detections = processor.process_image(image_np)
                        total_persons += persons_count
                        
                        # Convert back to RGB for display
                        if len(result_img.shape) == 3 and result_img.shape[2] == 3:
                            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                        
                        # Store result
                        processed_images.append({
                            'filename': uploaded_file.name,
                            'original': image_np,
                            'processed': result_img,
                            'persons_count': persons_count,
                            'detections': detections
                        })
                        
                        st.session_state.processed_files.append(uploaded_file.name)
                    
                    elif is_video:
                        st.warning(f"⚠️ Video processing for {uploaded_file.name} is available in the full version")
                
                status_text.text("✅ Processing complete!")
                progress_bar.progress(1.0)
                
                # Show results summary
                with results_container:
                    st.success(f"""
                    ### 🎉 Processing Complete!
                    - **Files Processed:** {len(processed_images)}
                    - **Total Persons Detected:** {total_persons}
                    - **Average Persons per Image:** {total_persons/len(processed_images):.1f}
                    """)
                    
                    # Create download button for all processed images
                    if processed_images:
                        # Create a zip file with all processed images
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            for img_data in processed_images:
                                # Convert processed image to bytes
                                img_pil = Image.fromarray(img_data['processed'])
                                img_bytes = io.BytesIO()
                                img_pil.save(img_bytes, format='PNG')
                                
                                # Add to zip
                                zip_file.writestr(
                                    f"blurred_{img_data['filename']}", 
                                    img_bytes.getvalue()
                                )
                        
                        st.download_button(
                            label="⬇️ Download All Processed Images (ZIP)",
                            data=zip_buffer.getvalue(),
                            file_name=f"blurred_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                            mime="application/zip"
                        )
                    
                    # Store results in session state for preview tab
                    st.session_state.last_results = processed_images
    
    with tab2:
        st.header("Preview Results")
        
        if hasattr(st.session_state, 'last_results') and st.session_state.last_results:
            for idx, img_data in enumerate(st.session_state.last_results):
                with st.expander(f"📸 {img_data['filename']} - {img_data['persons_count']} person(s) detected"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Original")
                        # Draw detection boxes on original
                        vis_img = img_data['original'].copy()
                        if len(vis_img.shape) == 3 and vis_img.shape[2] == 3:
                            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
                        
                        for detection in img_data['detections']:
                            x1, y1, x2, y2 = detection['bbox']
                            bx1, by1, bx2, by2 = detection['blur_region']
                            # Draw person box in blue
                            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            # Draw blur region in green
                            cv2.rectangle(vis_img, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                        
                        st.image(vis_img, use_column_width=True)
                    
                    with col2:
                        st.subheader("Processed")
                        st.image(img_data['processed'], use_column_width=True)
                    
                    # Download individual image
                    img_pil = Image.fromarray(img_data['processed'])
                    buf = io.BytesIO()
                    img_pil.save(buf, format='PNG')
                    
                    st.download_button(
                        label=f"⬇️ Download {img_data['filename']}",
                        data=buf.getvalue(),
                        file_name=f"blurred_{img_data['filename']}",
                        mime="image/png",
                        key=f"download_{idx}"
                    )
        else:
            st.info("👆 Process some images first to see results here")
    
    with tab3:
        st.header("📚 How to Use")
        
        st.markdown("""
        ### 🚀 Quick Start
        1. **Upload Files**: Click 'Browse files' or drag & drop your images
        2. **Configure Settings**: Use the sidebar to adjust blur region and intensity
        3. **Process**: Click the 'Start Processing' button
        4. **Download**: Get your privacy-compliant images!
        
        ### ⚙️ Settings Explained
        - **Blur Region**:
          - *Head/Helmet Only*: Blurs top 30% of detected person (recommended)
          - *Upper Body*: Blurs top 50% of detected person
          - *Full Person*: Blurs entire detected person
        
        - **Blur Intensity**: Higher values create stronger blur effect
        
        ### 🎯 Features
        - ✅ Automatic person detection using AI
        - ✅ Preserves image quality outside blur zones
        - ✅ Batch processing support
        - ✅ Preview before downloading
        - ✅ No data stored - all processing happens in your browser session
        
        ### 📋 Supported Formats
        - **Images**: PNG, JPG, JPEG, BMP
        - **Videos**: Coming soon!
        
        ### 🔒 Privacy Note
        All processing happens locally in your session. No images are stored on our servers.
        """)

if __name__ == "__main__":
    main()
