import argparse
import os
from imagegen import NoiseVisualizer
from utils import create_mp4_from_pil_images
import torch
import time

def main(song, output_path, seed, hop_length, distance, base_prompt, target_prompts, alpha, 
         guidance_scale, decay_rate, boost_factor, noteType, boost_threshold):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    visualizer = NoiseVisualizer(device=device, seed=seed)

    start_time = time.time()  # Start total timing
    visualizer.loadSong(song, hop_length=hop_length)
    print(f"Loaded song in {time.time() - start_time:.2f} seconds.")

    step_time = time.time()  # Start timing for each step
    print("Getting beat latents")
    latents = visualizer.getBeatLatents(distance=distance, noteType=noteType)
    print(f"Got beat latents in {time.time() - step_time:.2f} seconds.")
    
    fps = visualizer.getFPS()

    step_time = time.time()  # Reset step timing
    print("Getting prompt embeds")
    prompt_embeds = visualizer.getPromptEmbeds(basePrompt=base_prompt, 
                                               targetPromptChromaScale=target_prompts, 
                                               method="slerp", 
                                               alpha=alpha,
                                               decay_rate=decay_rate,
                                               boost_factor=boost_factor,
                                               boost_threshold=boost_threshold)
    print(f"Got prompt embeds in {time.time() - step_time:.2f} seconds.")

    step_time = time.time()  # Reset step timing
    images = visualizer.getVisuals(latents=latents, 
                                   promptEmbeds=prompt_embeds, 
                                   guidance_scale=guidance_scale)
    print(f"Generated visuals in {time.time() - step_time:.2f} seconds.")

    step_time = time.time()  # Reset step timing
    create_mp4_from_pil_images(image_array=images, 
                               output_path=output_path, 
                               song=song, 
                               fps=fps)
    print(f"Created MP4 in {time.time() - step_time:.2f} seconds.")
    
    print(f"Total execution time: {time.time() - start_time:.2f} seconds.")

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
    parser.add_argument("--guidance_scale", type=float, default=0, help="Guidance scale for image generation.")
    parser.add_argument("--decay_rate", type=float, default=0.8 )
    parser.add_argument("--boost_factor", type=float, default=1.75 )
    parser.add_argument("--boost_threshold", type=float, default=0.4 )
    parser.add_argument("--note_type", type=str, default="quarter",help="whole, half, or quarter" )
    

    args = parser.parse_args()

    main(song=args.song, 
         output_path=args.output, 
         seed=args.seed, 
         hop_length=args.hop_length, 
         distance=args.distance, 
         base_prompt=args.base_prompt, 
         target_prompts=args.target_prompts, 
         alpha=args.alpha, 
         guidance_scale=args.guidance_scale,
         decay_rate = args.decay_rate,
         boost_factor = args.boost_factor,
         boost_threshold = args.boost_threshold,
         noteType = args.note_type)