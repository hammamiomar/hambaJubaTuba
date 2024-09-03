import argparse
import os
from imagegen import NoiseVisualizer
from utils import create_mp4_from_pil_images
import torch

def main(song, output_path, seed, hop_length, distance, base_prompt, target_prompts, alpha, guidance_scale):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    visualizer = NoiseVisualizer(device=device, seed=seed)

    visualizer.loadSong(song, hop_length=hop_length)

    latents = visualizer.getSpecCircleLatents(distance=distance)
    fps = visualizer.getFPS()

    prompt_embeds = visualizer.getPromptEmbeds(basePrompt=base_prompt, 
                                               targetPromptChromaScale=target_prompts, 
                                               method="slerp", 
                                               alpha=alpha)

    images = visualizer.getVisuals(latents=latents, 
                                   promptEmbeds=prompt_embeds, 
                                   guidance_scale=guidance_scale)

    create_mp4_from_pil_images(image_array=images, 
                               output_path=output_path, 
                               song=song, 
                               fps=fps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a visualized video based on music and prompts.")
    parser.add_argument("--song", type=str, required=True, help="Path to the song file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output video.")
    parser.add_argument("--seed", type=int, default=133780085, help="Seed for noise generation.")
    parser.add_argument("--hop_length", type=int, default=377, help="Hop length for audio processing.")
    parser.add_argument("--distance", type=float, default=0.3, help="Distance for latent space generation.")
    parser.add_argument("--base_prompt", type=str, default="An octopus dancing with cigarettes", help="Base prompt for image generation.")
    parser.add_argument("--target_prompts", type=str, nargs='+', default=[
                           "what the dog doin.", 
                           "giant centipede.", 
                           "demon scary blood ", 
                           "man in a suit with juice", 
                           "massive hamburger yummmm",
                           "“broooo thers a beautiful chair", 
                           "hairy toes hospital", 
                           "car leaking water flooded", 
                           "mirror beautiful woman", 
                           "stop the cap now with a large coffee", 
                           "turkish rug in a room", 
                           "horse"], help="List of target prompts for chroma scaling.")
    parser.add_argument("--alpha", type=float, default=0.8, help="Alpha value for prompt interpolation.")
    parser.add_argument("--guidance_scale", type=float, default=0.2, help="Guidance scale for image generation.")
    #parser.add_argument("--decay_rate", type=float, default=0.8 ) #TODO FIX

    args = parser.parse_args()

    main(song=args.song, 
         output_path=args.output, 
         seed=args.seed, 
         hop_length=args.hop_length, 
         distance=args.distance, 
         base_prompt=args.base_prompt, 
         target_prompts=args.target_prompts, 
         alpha=args.alpha, 
         guidance_scale=args.guidance_scale)