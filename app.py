import os
import cv2
import sys
import torch
import tempfile
import streamlit as st
from pathlib import Path
from moviepy.editor import VideoFileClip, AudioFileClip
sys.path.append(str(Path(__file__).parent))
from inference_gfpgan import main as gfpgan_process

def extract_frames(video_path, output_dir, skip_frames=0):
    """Extract frames from video with optional skipping"""
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % (skip_frames + 1) == 0:
            cv2.imwrite(f"{output_dir}/frame_{frame_count:04d}.jpg", frame)
        frame_count += 1
    
    cap.release()
    print(f"Extracted {frame_count} frames (saved every {skip_frames+1} frames).")
    return frame_count, original_fps

def reconstruct_video(frame_dir, output_path, original_fps):
    """Reconstruct video from processed frames with original FPS"""
    frame_paths = sorted([f for f in os.listdir(frame_dir) if f.endswith(".jpg")])
    
    if not frame_paths:
        raise ValueError("No frames found for reconstruction")
    
    frame = cv2.imread(os.path.join(frame_dir, frame_paths[0]))
    h, w, _ = frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, original_fps, (w, h))
    
    for frame_name in frame_paths:
        frame = cv2.imread(os.path.join(frame_dir, frame_name))
        out.write(frame)
    
    out.release()

def add_audio_to_video(original_video_path, enhanced_video_path, output_path):
    """Add original audio to enhanced video using moviepy"""
    original_clip = VideoFileClip(original_video_path)
    enhanced_clip = VideoFileClip(enhanced_video_path)
    
    final_audio = original_clip.audio.set_duration(enhanced_clip.duration)
    final_clip = enhanced_clip.set_audio(final_audio)
    final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
    
    original_clip.close()
    enhanced_clip.close()
    final_clip.close()

def main():
    st.title("Video Enhancement with GFPGAN + MediaPipe")
    
    # Initialize session state for video persistence
    if 'enhanced_video_bytes' not in st.session_state:
        st.session_state.enhanced_video_bytes = None
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Video", type=["mp4"], key="video_uploader")
    
    with st.sidebar:
        st.header("Settings")
        skip_frames = st.slider("Skip Frames", 0, 5, 0)
        tile_size = st.slider("Tile Size", 256, 512, 400)
        version = st.selectbox("Model Version", ["1.3"])
        enhance_weight = st.slider("Enhance Weight", 0.1, 1.0, 0.5)
    
    if uploaded_file:        
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = os.path.join(tmp_dir, "input.mp4")
            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

                st.subheader("Original Video")
                st.video(input_path)
                
                if st.button("Enhance Video"):
                    with st.spinner("Processing..."):
                        try:
                            # Clear previous results
                            st.session_state.enhanced_video_bytes = None
                            st.session_state.show_results = False
                            
                            # Frame extraction
                            frame_dir = os.path.join(tmp_dir, "frames")
                            os.makedirs(frame_dir, exist_ok=True)
                            frame_count, original_fps = extract_frames(input_path, frame_dir, skip_frames)
                            st.info(f"Extracted {frame_count} frames")

                            # GFPGAN processing
                            results_dir = os.path.join(tmp_dir, "results")
                            os.makedirs(results_dir, exist_ok=True)
                        
                            args_list = [
                                "--input", frame_dir,
                                "--output", results_dir,
                                "--version", version,
                                "--upscale", "2",
                                "--weight", str(enhance_weight),
                                "--bg_tile", str(tile_size)
                            ]
                            gfpgan_process(args_list)
                        
                            # Video reconstruction
                            enhanced_video_path = os.path.join(tmp_dir, "enhanced_no_audio.mp4")
                            restored_dir = os.path.join(results_dir, "restored_imgs")
                            
                            if not os.path.exists(restored_dir):
                                restored_dir = results_dir
                            
                            reconstruct_video(restored_dir, enhanced_video_path, original_fps)
                            
                            # Add original audio
                            final_output_path = os.path.join(tmp_dir, "enhanced_final.mp4")
                            add_audio_to_video(input_path, enhanced_video_path, final_output_path)
                            
                            # Store the enhanced video in session state
                            with open(final_output_path, "rb") as f:
                                st.session_state.enhanced_video_bytes = f.read()
                            
                            st.session_state.show_results = True
                            st.success("Video enhancement completed!")
                        
                        except Exception as e:
                            st.error(f"Processing failed: {str(e)}")
                        finally:
                            torch.cuda.empty_cache()

    # Display enhanced video persistently
    if st.session_state.show_results and st.session_state.enhanced_video_bytes:
        st.subheader("Enhanced Video")
        st.video(st.session_state.enhanced_video_bytes)
        
        st.download_button(
            label="Download Enhanced Video",
            data=st.session_state.enhanced_video_bytes,
            file_name="enhanced_video.mp4",
            mime="video/mp4"
        )

        if st.button("Clear Results"):
            st.session_state.enhanced_video_bytes = None
            st.session_state.show_results = False
            st.experimental_rerun()

if __name__ == "__main__":
    main()

# import os
# import cv2
# import sys
# import torch
# import tempfile
# import streamlit as st
# from pathlib import Path
# from moviepy.editor import VideoFileClip, AudioFileClip
# import numpy as np
# sys.path.append(str(Path(__file__).parent))
# from inference_gfpgan import main as gfpgan_process

# def extract_frames(video_path, output_dir, skip_frames):
#     """Extract frames from video with optional skipping"""
#     cap = cv2.VideoCapture(video_path)
#     original_fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_count = 0
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if frame_count % (skip_frames + 1) == 0:
#             cv2.imwrite(f"{output_dir}/frame_{frame_count:04d}.jpg", frame)
#         frame_count += 1
    
#     cap.release()
#     print(f"Extracted {frame_count} frames (saved every {skip_frames+1} frames).")
#     return frame_count, original_fps  # Return original FPS

# def reconstruct_video(frame_dir, output_path, original_fps):
#     """Reconstruct video from processed frames with original FPS"""
#     frame_paths = sorted([f for f in os.listdir(frame_dir) if f.endswith(".jpg")])
    
#     if not frame_paths:
#         raise ValueError("No frames found for reconstruction")
    
#     frame = cv2.imread(os.path.join(frame_dir, frame_paths[0]))
#     h, w, _ = frame.shape
    
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, original_fps, (w, h))  # Use original FPS
    
#     for frame_name in frame_paths:
#         frame = cv2.imread(os.path.join(frame_dir, frame_name))
#         out.write(frame)
    
#     out.release()

# def add_audio_to_video(original_video_path, enhanced_video_path, output_path):
#     """Add original audio to enhanced video using moviepy"""
#     original_clip = VideoFileClip(original_video_path)
#     enhanced_clip = VideoFileClip(enhanced_video_path)
    
#     # Match audio duration with video duration
#     final_audio = original_clip.audio.set_duration(enhanced_clip.duration)
    
#     # Set enhanced clip's audio
#     final_clip = enhanced_clip.set_audio(final_audio)
#     final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
    
#     # Close clips to release resources
#     original_clip.close()
#     enhanced_clip.close()
#     final_clip.close()

# def main():
#     st.title("Video Enhancement with GFPGAN + MediaPipe")
    
#     # File uploader
#     uploaded_file = st.file_uploader("Upload Video", type=["mp4"])
#     with st.sidebar:
#         st.header("Settings")
#         skip_frames = st.slider("Skip Frames", 0, 5, 0)
#         tile_size = st.slider("Tile Size", 256, 512, 400)
#         version = st.selectbox("Model Version", ["1.3"])
#         enhance_weight = st.slider("Enhance Weight", 0.1, 1.0, 0.5)
    
#     if uploaded_file:        
#         with tempfile.TemporaryDirectory() as tmp_dir:
#             input_path = os.path.join(tmp_dir, "input.mp4")
#             with open(input_path, "wb") as f:
#                 f.write(uploaded_file.getbuffer())

#                 st.subheader("Original Video")
#                 st.video(input_path)
                
#                 if st.button("Enhance Video"):
#                     with st.spinner("Processing..."):
#                         try:
#                             # Step 1: Frame extraction
#                             frame_dir = os.path.join(tmp_dir, "frames")
#                             os.makedirs(frame_dir, exist_ok=True)
#                             frame_count, original_fps = extract_frames(input_path, frame_dir, skip_frames)
#                             st.info(f"Extracted {frame_count} frames")

#                             # Step 2: GFPGAN processing
#                             results_dir = os.path.join(tmp_dir, "results")
#                             os.makedirs(results_dir, exist_ok=True)
                        
#                             args_list = [
#                                 "--input", frame_dir,
#                                 "--output", results_dir,
#                                 "--version", version,
#                                 "--upscale", "2",
#                                 "--weight", str(enhance_weight),
#                                 "--bg_tile", str(tile_size)
#                             ]
#                             gfpgan_process(args_list)
                        
#                             # Step 3: Video reconstruction
#                             enhanced_video_path = os.path.join(tmp_dir, "enhanced_no_audio.mp4")
#                             restored_dir = os.path.join(results_dir, "restored_imgs")
                            
#                             if not os.path.exists(restored_dir):
#                                 restored_dir = results_dir
                            
#                             reconstruct_video(restored_dir, enhanced_video_path, original_fps)
                            
#                             # Step 4: Add original audio
#                             final_output_path = os.path.join(tmp_dir, "enhanced_final.mp4")
#                             add_audio_to_video(input_path, enhanced_video_path, final_output_path)
                            
#                             # Read final video into memory
#                             with open(final_output_path, "rb") as f:
#                                 video_bytes = f.read()
                            
#                             # Persistent video display and download
#                             st.subheader("Enhanced Video")
#                             st.video(video_bytes)
                            
#                             st.download_button(
#                                 label="Download Enhanced Video",
#                                 data=video_bytes,
#                                 file_name="enhanced_video.mp4",
#                                 mime="video/mp4"
#                             )
                        
#                         except Exception as e:
#                             st.error(f"Processing failed: {str(e)}")
#                         finally:
#                             torch.cuda.empty_cache()

# if __name__ == "__main__":
#     main()
