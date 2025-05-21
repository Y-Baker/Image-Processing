import streamlit as st
from core.manager import FilterManager
from utils.image_utils import convert_bgr_to_rgb, convert_rgb_to_bgr

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import json

filters = FilterManager()

# Initialize session state
if 'filters' not in st.session_state:
    st.session_state.filters = filters

if 'current_image' not in st.session_state:
    st.session_state.current_image = None

if 'original_image' not in st.session_state:
    st.session_state.original_image = None

if 'filter_history' not in st.session_state:
    st.session_state.filter_history = []

# Page configuration
st.set_page_config(
    page_title="Image Processing Toolbox",
    page_icon="🖼️",
    layout="wide"
)

# Title and description
st.title("🖼️ Image Processing Toolbox")
st.write("**Developed by Yousef Bakier**")
st.markdown("---")

# Sidebar for image upload and controls
with st.sidebar:
    st.header("📁 Image Upload")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        accept_multiple_files=False,
        help="Upload an image • File Limit: 100MB",
    )
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        # Convert to BGR for OpenCV processing
        if len(image_array.shape) == 3:
            image_bgr = convert_rgb_to_bgr(image_array)
        else:
            image_bgr = image_array
        
        # Store in session state
        if st.session_state.current_image is None:
            st.session_state.original_image = image_bgr.copy()
            st.session_state.current_image = image_bgr.copy()
        
        st.success("Image loaded successfully!")
        st.image(image, caption="Uploaded Image", width=250)
    
    # Reset button
    if st.button("🔄 Reset to Original", key="reset_btn"):
        if st.session_state.original_image is not None:
            st.session_state.current_image = st.session_state.original_image.copy()
            st.session_state.filter_history = []
            st.rerun()
    
    # Clear all 
    if st.button("🗑️ Clear All", key="clear_btn"):
        st.session_state.original_image = None
        st.session_state.current_image = None
        st.session_state.filter_history = []
        st.rerun()

# Main content area
if st.session_state.current_image is not None:
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["🔧 Filter Application", "📊 Filter History", "💾 Download"])
    
    with tab1:
        # Split into two columns
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🛠️ Filter Selection")
            
            # Get all services
            services = st.session_state.filters.get_all_services()
            
            # Service category selection
            service_category = st.selectbox(
                "Select Filter Category",
                list(services.keys()),
                key="service_select"
            )
            
            # Operation selection
            if service_category:
                operations = list(services[service_category].keys())
                operation = st.selectbox(
                    "Select Operation",
                    operations,
                    key="operation_select"
                )
                
                # Parameter inputs
                if operation:
                    operation_info = services[service_category][operation]
                    params = {}
                    
                    st.write("**Parameters:**")
                    if operation_info.get("params"):
                        step_config = operation_info.get("step", {})  # Optional per-param step sizes

                        for i, param in enumerate(operation_info["params"]):
                            param_label = param.replace('_', ' ').title()

                            if "options" in operation_info:
                                # Dropdown for categorical options
                                params[param] = st.selectbox(
                                    param_label,
                                    operation_info["options"],
                                    key=f"param_{param}"
                                )

                            elif "range" in operation_info:
                                # Handle single or multi-param ranges
                                param_range = operation_info["range"]
                                if isinstance(param_range[0], list):
                                    param_range = param_range[i]

                                # Infer type and set step
                                is_float = isinstance(param_range[0], float)
                                step = step_config.get(param, 0.1 if is_float else 1)

                                # Ensure default value is valid
                                default_value = float((param_range[0] + param_range[1]) / 2) if is_float else (param_range[0] + param_range[1]) // 2

                                # Ensure default is odd if step = 2 (for kernel sizes like in Median Filter)
                                if step == 2 and isinstance(default_value, int) and default_value % 2 == 0:
                                    default_value += 1

                                # Create slider
                                params[param] = st.slider(
                                    param_label,
                                    min_value=param_range[0],
                                    max_value=param_range[1],
                                    value=default_value,
                                    step=step,
                                    key=f"param_{param}"
                                )

                            else:
                                # Fallback to text input
                                params[param] = st.text_input(
                                    param_label,
                                    key=f"param_{param}"
                                )
                    else:
                        st.write("No parameters required for this operation.")
                    
                    if operation == "Show Histogram":
                        show_hist_button = st.button("📊 Show Histogram", key="show_histogram_btn")
                        
                        if show_hist_button:
                            try:
                                with st.spinner("Generating histogram..."):
                                    hist_data = st.session_state.filters.process_image(
                                        st.session_state.current_image,
                                        service_category,
                                        operation,
                                        params if params else None
                                    )

                                    # Show histogram in an expander or modal-like area
                                    with st.expander("📈 Histogram Result", expanded=True):
                                        import matplotlib.pyplot as plt
                                        fig, ax = plt.subplots()

                                        if 'gray' in hist_data:
                                            ax.plot(hist_data['gray'], color='black')
                                            ax.set_title("Grayscale Histogram")
                                        else:
                                            for color, hist in hist_data.items():
                                                ax.plot(hist, color=color)
                                            ax.set_title("RGB Histogram")
                                        
                                        st.pyplot(fig)
                                        
                                        # Confirmation button to collapse/hide it
                                        if st.button("✅ OK", key="close_histogram"):
                                            st.rerun()

                            except Exception as e:
                                st.error(f"Error showing histogram: {str(e)}")
                    else:
                        if st.button("✨ Apply Filter", key="apply_filter"):
                            try:
                                with st.spinner("Processing image..."):
                                    processed_image = st.session_state.filters.process_image(
                                        st.session_state.current_image,
                                        service_category,
                                        operation,
                                        params if params else None
                                    )
                                    
                                    # Update current image
                                    st.session_state.current_image = processed_image
                                    
                                    # Add to history
                                    st.session_state.filter_history.append({
                                        "category": service_category,
                                        "operation": operation,
                                        "params": params.copy() if params else {}
                                    })
                                    
                                    st.success(f"Applied {operation} successfully!")
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Error applying filter: {str(e)}")
        
        with col2:
            st.subheader("🖼️ Image Preview")
            
            # Display current image
            if st.session_state.current_image is not None:
                # Convert BGR to RGB for display
                display_image = convert_bgr_to_rgb(st.session_state.current_image)
                
                # Create comparison view
                if st.session_state.original_image is not None:
                    col_orig, col_current = st.columns(2)
                    
                    with col_orig:
                        st.write("**Original Image**")
                        original_display = convert_bgr_to_rgb(st.session_state.original_image)
                        st.image(original_display, use_container_width=True)
                    
                    with col_current:
                        st.write("**Current Image**")
                        st.image(display_image, use_container_width=True)
                else:
                    st.image(display_image, use_container_width=True)
                
                # Image information
                height, width = st.session_state.current_image.shape[:2]
                channels = st.session_state.current_image.shape[2] if len(st.session_state.current_image.shape) == 3 else 1
                
                st.info(f"**Image Info:** {width}x{height} pixels, {channels} channel(s)")
    
    with tab2:
        st.subheader("📝 Applied Filters History")
        
        if st.session_state.filter_history:
            for i, filter_info in enumerate(st.session_state.filter_history):
                with st.expander(f"Filter {i+1}: {filter_info['operation']}", expanded=False):
                    st.write(f"**Category:** {filter_info['category']}")
                    st.write(f"**Operation:** {filter_info['operation']}")
                    if filter_info['params']:
                        st.write("**Parameters:**")
                        for param, value in filter_info['params'].items():
                            st.write(f"  - {param}: {value}")
                    else:
                        st.write("**Parameters:** None")
            
            # Export filter sequence
            if st.button("📋 Export Filter Sequence"):
                filter_sequence = json.dumps(st.session_state.filter_history, indent=2)
                st.text_area("Filter Sequence (JSON)", filter_sequence, height=200)
        else:
            st.info("No filters applied yet.")
    
    with tab3:
        st.subheader("💾 Download Processed Image")
        
        if st.session_state.current_image is not None:
            # Convert to PIL for saving
            display_image = convert_bgr_to_rgb(st.session_state.current_image)
            pil_image = Image.fromarray(display_image)
            
            # Create download button
            buf = BytesIO()
            pil_image.save(buf, format='PNG')
            
            st.download_button(
                label="📥 Download PNG",
                data=buf.getvalue(),
                file_name="processed_image.png",
                mime="image/png"
            )
            
            # Also offer JPEG option
            buf_jpg = BytesIO()
            if pil_image.mode == 'RGBA':
                pil_image = pil_image.convert('RGB')
            pil_image.save(buf_jpg, format='JPEG', quality=95)
            
            st.download_button(
                label="📥 Download JPG",
                data=buf_jpg.getvalue(),
                file_name="processed_image.jpg",
                mime="image/jpeg"
            )
        else:
            st.info("No processed image available for download.")

else:
    st.markdown("""
    <div style='text-align: center; margin-top: 50px;'>
        <h2>🚀 Welcome to the Image Processing Toolbox!</h2>
        <p style="font-size: 18px; color: #555;">
            Start by uploading an image from the sidebar to explore a wide range of filters and operations.<br>
            Enhance, transform, and analyze your images with ease.
        </p>
        <img src="https://github.com/Y-Baker/Image-Processing/main/assets/upload.png"
             alt=""
             style="margin-top: 30px; border-radius: 15px; max-width: 350px; box-shadow: 0 4px 16px rgba(0,0,0,0.08); display: none;"
             onerror="this.style.display='none';">
        <p style="margin-top: 20px; color: #888;">No image uploaded yet.</p>
    </div>
    """,
    unsafe_allow_html=True
    )

# Footer
st.markdown("---")
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 10px 0 5px 0;
        font-size: 15px;
        z-index: 100;
        border-top: 1px solid #e6e6e6;
    }
    </style>
    <div class="footer">
        <p>🖼️ Image Processing Toolbox | Developed by <strong>Yousef Bakier</strong></p>
        <p><em>© 2025 Yousef Bakier</em></p>
    </div>
    """,
    unsafe_allow_html=True
)
