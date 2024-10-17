import argparse
import os
from src.imagegen import NoiseVisualizer
from src.utils import create_mp4_from_pil_images
import torch
import time

dtype_map = {
    'float32': torch.float32,
    'float64': torch.float64,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
    'int32': torch.int32,
    'int64': torch.int64
}

def main(song, output_path, device, weightType, seed, hop_length, distance, base_prompt, target_prompts, alpha, 
         noteType, sigma_time, sigma_chroma, jitter_strength, number_of_chromas, embed_type,number_of_chromas_focus, bpm,
         num_prompt_shuffles, guidance_scale, num_inference_steps, batch_size):
    
    visualizer = NoiseVisualizer(device=device, weightType=weightType, seed=seed)
    visualizer.loadPipeSdXL()
    print(f'loaded {device} as device')
    start_time = time.time()  # Start total timing
    visualizer.loadSong(song, hop_length=hop_length, number_of_chromas=number_of_chromas, bpm=bpm)
    print(f"Loaded song in {time.time() - start_time:.2f} seconds.")

    step_time = time.time()  # Start timing for each step
    print("Getting beat latents")
    latents = visualizer.getBeatLatentsCircle(noteType=noteType, distance=distance, height=1024, width=1024)
    #latents = visualizer.getBeatLatentsCircle(distance=distance, noteType=noteType, jitter_strength=jitter_strength)
    #latents = visualizer.getBeatLatentsSpiral(distance=distance, noteType=noteType, spiral_rate=spiral_rate)
    print(f"Got beat latents in {time.time() - step_time:.2f} seconds.")
    
    fps = visualizer.getFPS()

    step_time = time.time()  # Reset step timing
    print("Getting prompt embeds")
    if embed_type == "focus":
        prompt_embeds,prompt_embeds_pooled = visualizer.getPromptEmbedsOnsetFocus(basePrompt=base_prompt, 
                                               targetPromptChromaScale=target_prompts, 
                                               alpha=alpha,
                                               sigma=sigma_time)
    else:
        prompt_embeds = visualizer.getPromptEmbedsCum(basePrompt=base_prompt, 
                                                targetPromptChromaScale=target_prompts, 
                                                alpha=alpha,
                                                sigma_time=sigma_time,
                                                sigma_chroma=sigma_chroma,
                                                number_of_chromas_focus=number_of_chromas_focus,
                                                num_prompt_shuffles=num_prompt_shuffles)
    
    
    print(f"Got prompt embeds in {time.time() - step_time:.2f} seconds.")

    step_time = time.time()  # Reset step timing
    images = visualizer.getVisualsPooled(latents=latents, 
                                   promptEmbeds=prompt_embeds,
                                   promptEmbedsPooled=prompt_embeds_pooled,
                                   num_inference_steps=num_inference_steps, 
                                   guidance_scale=guidance_scale,
                                   batch_size=batch_size)
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
    parser.add_argument("--device", type=str, required=True, help="device")
    parser.add_argument("--weightType", type=str, required=True, help="weighType")
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
    parser.add_argument("--sigma_time", type=float, default=2, help="Sigma value for prompt interpolation.")
    parser.add_argument("--sigma_chroma", type=float, default=1, help="Sigma value for prompt interpolation.")
    parser.add_argument("--note_type", type=str, default="quarter",help="whole, half, or quarter" )
    parser.add_argument("--jitter_strength", type=float, default=0.1, help="Jitter strength for latent interpolation")
    parser.add_argument("--number_of_chromas", type=int, default=12)
    parser.add_argument("--number_of_chromas_focus", type=int, default=6)
    parser.add_argument("--embed_type", type=str, default="focus")
    parser.add_argument("--bpm", type=int, default=None)
    parser.add_argument("--num_prompt_shuffles", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=8)

    

    args = parser.parse_args()

    main(song=args.song, 
         output_path=args.output, 
         device=args.device,
         weightType=dtype_map[args.weightType],
         seed=args.seed, 
         hop_length=args.hop_length, 
         distance=args.distance, 
         base_prompt=args.base_prompt, 
         target_prompts=args.target_prompts, 
         alpha=args.alpha, 
         noteType = args.note_type,
         sigma_time = args.sigma_time,
         sigma_chroma = args.sigma_chroma,
         jitter_strength = args.jitter_strength,
         number_of_chromas = args.number_of_chromas,
         number_of_chromas_focus = args.number_of_chromas_focus,
         embed_type = args.embed_type,
         bpm=args.bpm,
         num_prompt_shuffles = args.num_prompt_shuffles,
         num_inference_steps = args.num_inference_steps,
         guidance_scale = args.guidance_scale,
         batch_size=args.batch_size)