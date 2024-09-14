import torch
import numpy as np
import moviepy.editor as mpy
import os
import math

def LatentInitCircular(steps, distance, weightType=torch.float16): # only circular walk for now..
        shape = (1, 4, 64, 64) # batch, latent channnels, height, width
        
        walkNoiseX = torch.randn(shape,dtype=weightType)
        walkNoiseY = torch.randn(shape,dtype=weightType)
        
        walkScaleX = torch.cos(torch.linspace(0, distance, steps, dtype=weightType) * math.pi)
        walkScaleY = torch.sin(torch.linspace(0, distance, steps, dtype=weightType) * math.pi)
        
        noiseX = torch.tensordot(walkScaleX, walkNoiseX, dims=0)
        noiseY = torch.tensordot(walkScaleY, walkNoiseY, dims=0)
        
        latents = torch.add(noiseX, noiseY)
        
        return latents
    
# def create_mp4_from_pil_images(image_array, output_path, song, fps):
#     """
#     Creates an MP4 video at the specified frame rate from an array of PIL images.

#     :param image_array: List of PIL images to be used as frames in the video.
#     :param output_path: Path where the output MP4 file will be saved.
#     :param fps: Frames per second for the output video. Default is 60.
#     """
#     # Convert PIL images to moviepy's ImageClip format
#     clips = [mpy.ImageClip(np.array(img)).set_duration(1/fps) for img in image_array]
    
#     # Concatenate all the clips into a single video clip
#     video = mpy.concatenate_videoclips(clips, method="compose")
    
#     video = video.set_audio(mpy.AudioFileClip(song, fps=44100))
#     # Write the result to a file
#     video.write_videofile(output_path, fps=fps, audio_codec='aac')

def create_mp4_from_pil_images(image_array, output_path, song, fps, device='cpu'):
    """
    Creates an MP4 video at the specified frame rate from an array of PIL images.
    Optionally utilizes NVIDIA's CUDA for faster encoding if device is set to 'cuda'.

    :param image_array: List of PIL images to be used as frames in the video.
    :param output_path: Path where the output MP4 file will be saved.
    :param song: Path to the audio file to be added to the video.
    :param fps: Frames per second for the output video.
    :param device: Encoding device to use ('cuda' for GPU acceleration, 'cpu' for default encoding).
    """
    # Validate the device parameter
    if device not in ['cuda', 'cpu', 'mps']:
            raise ValueError("Invalid device. Choose 'cuda' for GPU acceleration or 'cpu' for default encoding.")
    
    if device == "cuda":
        ffmpeg_path = '/usr/bin/ffmpeg'  # Example path
        os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"


    # Convert PIL images to MoviePy ImageClips
    clips = [
        mpy.ImageClip(np.array(img))
        .set_duration(1 / fps)
        for img in image_array
    ]

    # Concatenate all the clips into a single video clip
    video = mpy.concatenate_videoclips(clips, method="compose")

    # Add audio to the video
    video = video.set_audio(mpy.AudioFileClip(song, fps=44100))

    # Configure codec and ffmpeg parameters based on the device
    if device == 'cuda':
        codec = 'h264_nvenc'  # NVIDIA's hardware-accelerated H.264 encoder
        ffmpeg_params = [
            '-preset', 'fast',        # Encoding speed/quality trade-off
            '-b:v', '5M',             # Video bitrate (adjust as needed)
            '-maxrate', '5M',         # Maximum bitrate
            '-bufsize', '10M',        # Rate control buffer size
            '-pix_fmt', 'yuv420p'     # Pixel format for compatibility
        ]
    else:
        codec = 'libx264'           # Default CPU-based H.264 encoder
        ffmpeg_params = [
            '-preset', 'medium',      # Encoding speed/quality trade-off
            '-crf', '23',             # Constant Rate Factor for quality control
            '-pix_fmt', 'yuv420p'     # Pixel format for compatibility
        ]

    # Write the video file with the specified codec and parameters
    video.write_videofile(
        output_path,
        fps=fps,
        codec=codec,
        audio_codec='aac',
        ffmpeg_params=ffmpeg_params,
        threads=4,                # Adjust based on your CPU cores
        verbose=True,
        logger='bar'              # Progress bar
    )