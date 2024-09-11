# DiffusionMusicVisualizer




[![Sample](https://img.youtube.com/vi/a_UTZs5M2ns/0.jpg)](https://www.youtube.com/watch?v=a_UTZs5M2ns)


A music visualizer which creates visualizations via stable diffusion XS and audio features via librosa.

Idea is that I can take music signals(tempo,pitch) and then have them follow a a 2d trajectory which translates to the latent space being modified in the stable diffusion inference, generating music visualizations as parts of the image shift according to different signals shifting in the song. 

SDXS is being used due to the 1 step generation of an image, hopefully leading to real time visualization. 

Currently, pitch interpolates the noise vector, allowing the image generated to modulate according to the beat. 

The detected notes in the song correspond to a chromatic scale of 12 prompts, which interpolate the base prompt, shifting the image generated according to what note is detected.

Future plan: Reduce spaziness of high bpm music, interpolation of overall contet of the music. Perhaps inpainting and changing different areas of the image according to different notes, such as bass affecting bottom of the picture, etc. 

Learning Goals: diffusion noise methods, quantization, audio transforms


## CURRENT IMPLEMENTATION

Noise vectors are made via latent walking in a circlular direction which is influenced by the beat and power of percussion. There is now cubic in out easing

Promt embeddings is a mess
## CURRENT TESTING
- Make prompt embedding interpolate with smoothing via onset and the chroma that is most influential to the onset.
- Switching to 

## TODO
- [X] Create circular UNet input noise walk
- [X] Create clip latent interpolation
- [X] Get signals from audio that can correspond with dominant melody, then get pitch into signal
- [X] Smooth out high bpm music
- [X] Switch to instaflow
- [ ] Real time performance (perhaps 1 step generation, or 1 step scheduler pass, or quantization, or mlx optimization... loading song first then playing it to simulate realtime)

## License

This project is licensed under the Non-Commercial License - see the [LICENSE](LICENSE) file for details.

Stable diffusion model of this project is from [Original Repository](https://github.com/IDKiro/sdxs) by [Original Author](https://github.com/IDKiro/), which is licensed under the Apache License. See the [LICENSE-MIT](LICENSE-MIT) file for details.
