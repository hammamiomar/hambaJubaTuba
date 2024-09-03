import torch
import numpy as np
import moviepy.editor as mpy
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
    
def create_mp4_from_pil_images(image_array, output_path, song, fps):
    """
    Creates an MP4 video at the specified frame rate from an array of PIL images.

    :param image_array: List of PIL images to be used as frames in the video.
    :param output_path: Path where the output MP4 file will be saved.
    :param fps: Frames per second for the output video. Default is 60.
    """
    # Convert PIL images to moviepy's ImageClip format
    clips = [mpy.ImageClip(np.array(img)).set_duration(1/fps) for img in image_array]
    
    # Concatenate all the clips into a single video clip
    video = mpy.concatenate_videoclips(clips, method="compose")
    
    video = video.set_audio(mpy.AudioFileClip(song, fps=44100))
    # Write the result to a file
    video.write_videofile(output_path, fps=fps, audio_codec='aac')
    