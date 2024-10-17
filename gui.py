import gradio as gr
import torch
import random

from main import main as mainAudio
from mainCtrl import main as mainCtrl


with gr.Blocks(fill_width=True, fill_height=True,theme=gr.themes.Monochrome()) as demo:
    gr.Markdown(
        """
        # Diffusion Music Visualizer
        """
    )
    # Input type define, so can get different parameters depending on type of input later on
    with gr.Row(equal_height=True):
        inputType = gr.Radio(["Audio","Video"],value="Audio",label="Making visualiation purely off audio, or with controlnet",scale=0)
        
        # @gr.render(inputs=inputType)
        # def uploadfile(inputType):
        #     if inputType == "Audio":
        #         filePath = gr.File(label="Upload Audio", type="filepath",file_types=["audio"])
        #     else:
        #         filepath = gr.File(label = "Upload Video",type="filepath",file_types=["video"])
        fileAudio = gr.File(label="Upload Audio", type="filepath", file_types=["audio"], visible=True)
        fileVideo = gr.File(label="Upload Video", type="filepath", file_types=["video"], visible=False)

        # Update visibility of file inputs based on inputType
        def update_file_input(inputType):
            if inputType == "Audio":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)

        inputType.change(fn=update_file_input, inputs=inputType, outputs=[fileAudio, fileVideo])
                
    # Defining the prompt scale
    basePrompt = gr.TextArea(label="Base Prompt for interpolation")
    gr.Markdown("## 12 Note Chromatic Prompt Scale, from low to high. 77 token limit.")
    with gr.Row():
        prompt1 = gr.TextArea(label="C")
        prompt2 = gr.TextArea(label="C-sharp")
        prompt3 = gr.TextArea(label="D")
        prompt4 = gr.TextArea(label="D-sharp")
        prompt5 = gr.TextArea(label="E")
        prompt6 = gr.TextArea(label="F")
    with gr.Row():
        prompt7 = gr.TextArea(label="F-sharp")
        prompt8 = gr.TextArea(label="G")
        prompt9 = gr.TextArea(label="G-sharp")
        prompt10 = gr.TextArea(label="A")
        prompt11 = gr.TextArea(label="A-sharp")
        prompt12 = gr.TextArea(label="B")
    gr.Markdown("# Music Processing Parameters")
    with gr.Row():
        with gr.Column():
            noteType = gr.Dropdown(["quarter","half","whole"],interactive=True,value="quarter",label="Downbeat of the noise interpolation.")
        
        with gr.Column():
            # Hop Length slider, visible only when inputType is "Audio"
            hopLength = gr.Slider(minimum=368, maximum=735, value=368, label="""Hop length for music analysis, 
                                    which also gives final FPS of video. Minimum hop length gives 60fps, max gives 30.""", visible=True)

        # Update visibility of hopLength based on inputType
        def update_hop_length_visibility(inputType):
            if inputType == "Audio":
                return gr.update(visible=True)
            else:
                return gr.update(visible=False)

        inputType.change(fn=update_hop_length_visibility, inputs=inputType, outputs=hopLength)
    
    gr.Markdown("# Visualization Parameters")
    with gr.Row():
        with gr.Column():
            interpolationType = gr.Radio(["Focus", "Cumulative"],value="Focus",interactive=True,
                                        label="""Focus is where prompt interpolation occurs between base prompt and
                                        a focused onset note prompt, whereas cumulative is a weighted sum of multiple
                                        prompt notes.""")
            noiseDistance = gr.Slider(minimum=0.001,maximum=2,interactive=True,value=0.1,label="Circular Noise Interpolation Distance")
            alpha = gr.Slider(minimum=0.1,maximum=1.0,interactive=True,value=0.8,label="Alpha Value for Prompt Interpolation")
            sigmaTime = gr.Slider(minimum=0.01,maximum=10.0,interactive=True,value=2,label="Sigma Value for Gaussian Smoothing Between Prompt Interpolation")
        
        with gr.Column():
            # Parameters specific to "Cumulative" interpolation type
            sigmaAlpha = gr.Slider(minimum=0.01, maximum=10.0, value=2, label="Sigma Value for Gaussian Smoothing Within Cumulative Prompt Interpolation", visible=False)
            numChromaFocus = gr.Slider(minimum=1, maximum=12, value=4, label="Number of chroma notes to influence each weighted sum", visible=False)
            # Update visibility of cumulative parameters based on interpolationType
            def update_cumulative_params(interpolationType):
                if interpolationType == "Cumulative":
                    return gr.update(visible=True), gr.update(visible=True)
                else:
                    return gr.update(visible=False), gr.update(visible=False)

            interpolationType.change(fn=update_cumulative_params, inputs=interpolationType, outputs=[sigmaAlpha, numChromaFocus])

    gr.Markdown("# Processing Parameters")
    with gr.Row():
        device = gr.Dropdown(["cuda","mps","cpu"],label="Device Type",interactive=True)
        weightType = gr.Dropdown(["float16","float32"],label="Weight Type",interactive=True)
        batchSize = gr.Slider(minimum=1,maximum=32,step=1,value=1,interactive=True,label="Batch Size for Generation")
        
        # ControlNet Type, visible only when inputType is "Video"
        ctrlType = gr.Radio(["Depth", "Pose"], label="ControlNet Type", visible=False)

        # Update visibility of ctrlType based on inputType
        def update_ctrl_type_visibility(inputType):
            if inputType == "Video":
                return gr.update(visible=True)
            else:
                return gr.update(visible=False)

        inputType.change(fn=update_ctrl_type_visibility, inputs=inputType, outputs=ctrlType)
        
        seed = gr.Number(label="Seed", value=133780085)
        generateSeedButton = gr.Button("Generate Random Seed")

        # Function to generate a random seed and update the seed input field
        def generate_random_seed():
            random_seed = random.randint(0, 2**32 - 1)
            return gr.update(value=random_seed)

        generateSeedButton.click(fn=generate_random_seed, outputs=seed)

    # Output Path input
    outputPath = gr.Textbox(label="Output Path", value="output.mp4")

    # Generate button and output status
    generateButton = gr.Button("Generate Visualization")
    output_status = gr.Textbox(label="Status", interactive=False)

    # Function to generate visualization
    def generate_visualization(inputType, fileAudio, fileVideo, basePrompt, prompt1, prompt2, prompt3, prompt4, prompt5, prompt6,
                               prompt7, prompt8, prompt9, prompt10, prompt11, prompt12,
                               noteType, hopLength, interpolationType, noiseDistance, alpha, sigmaTime,
                               sigmaAlpha, numChromaFocus, device, weightType, batchSize, ctrlType, seed, outputPath):
        # Collect the target prompts
        target_prompts = [prompt1, prompt2, prompt3, prompt4, prompt5, prompt6,
                          prompt7, prompt8, prompt9, prompt10, prompt11, prompt12]

        # Set the output path
        output_path = outputPath

        # Set default values for parameters not collected from UI
        seed = int(seed)
        distance = noiseDistance
        alpha = alpha
        sigma_time = sigmaTime
        sigma_chroma = sigmaAlpha if sigmaAlpha is not None else 1
        number_of_chromas_focus = numChromaFocus if numChromaFocus is not None else 4
        embed_type = interpolationType.lower()
        num_prompt_shuffles = 4
        num_inference_steps = 1
        guidance_scale = 0
        batch_size = int(batchSize)

        try:
            if inputType == "Audio":
                if fileAudio is None:
                    return "Please upload an audio file."
                song = fileAudio.name
                hop_length = int(hopLength)
                mainAudio(song=song, output_path=output_path, device=device, weightType=weightType, seed=seed,
                          hop_length=hop_length, distance=distance, base_prompt=basePrompt, target_prompts=target_prompts,
                          alpha=alpha, noteType=noteType, sigma_time=sigma_time, sigma_chroma=sigma_chroma,
                          embed_type=embed_type, number_of_chromas_focus=number_of_chromas_focus,
                          num_prompt_shuffles=num_prompt_shuffles, guidance_scale=guidance_scale,
                          num_inference_steps=num_inference_steps, batch_size=batch_size)
            else:
                if fileVideo is None:
                    return "Please upload a video file."
                song = fileVideo.name
                hop_length = int(hopLength) if hopLength is not None else 368  # Default value
                mainCtrl(song=song, output_path=output_path, device=device, weightType=weightType, seed=seed,
                         hop_length=hop_length, distance=distance, base_prompt=basePrompt, target_prompts=target_prompts,
                         alpha=alpha, noteType=noteType, sigma_time=sigma_time, sigma_chroma=sigma_chroma,embed_type=embed_type,
                         number_of_chromas_focus=number_of_chromas_focus, num_prompt_shuffles=num_prompt_shuffles,
                         guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, batch_size=batch_size,
                         ctrlType=ctrlType.lower())
            return "Visualization generated successfully!"
        except Exception as e:
            return f"An error occurred: {str(e)}"

    # List of inputs for the generate_visualization function
    inputs = [inputType, fileAudio, fileVideo, basePrompt, prompt1, prompt2, prompt3, prompt4, prompt5, prompt6,
              prompt7, prompt8, prompt9, prompt10, prompt11, prompt12,
              noteType, hopLength, interpolationType, noiseDistance, alpha, sigmaTime,
              sigmaAlpha, numChromaFocus, device, weightType, batchSize, ctrlType, seed, outputPath]

    generateButton.click(fn=generate_visualization, inputs=inputs, outputs=output_status)

# Enable queueing
demo.queue()
demo.launch()