import torch
from torch import FloatTensor, LongTensor, Tensor, Size, lerp, zeros_like
from torch.linalg import norm
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import gaussian_filter

from diffusers import StableDiffusionPipeline
import librosa
import tqdm as tqdm
from PIL import Image
import numpy as np
import moviepy.editor as mpy
import math




class NoiseVisualizer:
    def __init__(self, device="mps", weightType=torch.float16, seed=42069):
        torch.manual_seed(seed)
        self.device = device
        self.weightType = weightType
        self.pipe = StableDiffusionPipeline.from_pretrained("IDKiro/sdxs-512-dreamshaper", torch_dtype=weightType)
        self.textEncoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer
    
    def loadSong(self,file,hop_length, number_of_chromas):
        y, sr = librosa.load(file) # 3 min 52 sec
        self.hop_length=hop_length
        self.sr = sr
        self.y = y
        
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Latent generaton
        
        self.beat, self.beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr,hop_length=self.hop_length)
        
        melSpec = librosa.feature.melspectrogram(y=y_percussive,sr=sr, n_mels = 256, hop_length=self.hop_length)
        self.melSpec = np.sum(melSpec,axis=0)
        
        
        # Prompt Generation
        self.chroma_cq = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, hop_length=self.hop_length)
        
        # Onset Detection
        oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        onset_raw = librosa.onset.onset_detect(
            onset_envelope=oenv, backtrack=False, hop_length=self.hop_length
        )
        self.onset_bt = librosa.onset.onset_backtrack(onset_raw, oenv) # shape: (num_frames)

        # Chroma Delta Computation
        chroma_cq_delta = librosa.feature.delta(self.chroma_cq, order=1)
        chroma_cq_delta_abs = np.abs(chroma_cq_delta)

        self.top_chromaOnset = np.argmax(chroma_cq_delta, axis=0)
        # Get indices of the top N chromas with the largest deltas at each frame
        self.top_chromas = np.argsort(-chroma_cq_delta_abs, axis=0)[:number_of_chromas, :]  # shape: (number_of_chromas, numFrames)

        self.steps = self.melSpec.shape[0]
        self.number_of_chromas = number_of_chromas  # Store for use in getPromptEmbeds
        
    def getEasedBeats(self, noteType):  # for circle..
        def cubic_in_out_numpy(t):
            return np.where(t < 0.5, 4 * t**3, 1 - (-2 * t + 2)**3 / 2)

        if noteType == "half":
            beat_frames = self.beat_frames[::2]
        elif noteType == "whole":
            beat_frames = self.beat_frames[::4]
        else:
            beat_frames = self.beat_frames
            
        # Initialize output array
        output_array = np.zeros(self.steps, float)

        # Normalize the mel spectrogram values at the beat frames and set them in the output array
        output_array[beat_frames] = librosa.util.normalize(self.melSpec[beat_frames])

        # Interpolate between the beat frames
        last_beat_idx = None
        for current_idx in beat_frames:
            if last_beat_idx is not None:
                # Interpolate between the previous beat and current beat
                for j in range(last_beat_idx + 1, current_idx):
                    t = (j - last_beat_idx) / (current_idx - last_beat_idx)
                    eased_value = cubic_in_out_numpy(np.array(t))
                    output_array[j] = eased_value
            last_beat_idx = current_idx

        return torch.tensor(output_array, dtype=self.weightType)
    def getBeatLatentsCircle(self, distance, noteType, jitter_strength): 
        # shape: (batch, latent channels, height, width)
        shape = (1, 4, 64, 64)
        
        # Initialize random noise for X and Y coordinates of the walk
        walkNoiseX = torch.randn(shape, dtype=self.weightType, device=self.device)
        walkNoiseY = torch.randn(shape, dtype=self.weightType, device=self.device)
        
        # Apply the easing values to the scales (mapping easing values to noise)
        easing_values = self.getEasedBeats(noteType=noteType).to(self.device)
        
        # Compute cumulative sum of easing values to get the angle
        # Adjust the scaling factor to control speed of rotation
        scaling_factor = distance * 2 * math.pi  # Multiply by 2π for full rotations
        angles = torch.cumsum(easing_values, dim=0) * scaling_factor
        
        # # Optionally add some jitter to the angles to add randomness
        jitter = torch.randn_like(angles) * jitter_strength
        angles += jitter
        
        # Compute cos and sin of the angles
        walkScaleX = torch.cos(angles)
        walkScaleY = torch.sin(angles)
        
        # Expand walkScaleX and walkScaleY to match dimensions for multiplication
        walkScaleX = walkScaleX.view(-1, 1, 1, 1)
        walkScaleY = walkScaleY.view(-1, 1, 1, 1)
        
        # Generate the noise tensors based on the interpolated scales
        noiseX = walkScaleX * walkNoiseX
        noiseY = walkScaleY * walkNoiseY
        
        # Add the noise contributions for X and Y to create the latent walk
        latents = noiseX + noiseY  # Shape: (steps, 4, 64, 64)
        return latents
    
    def getBeatLatentsSpiral(self, distance, noteType, spiral_rate): 
        # shape: (batch, latent channels, height, width)
        shape = (1, 4, 64, 64)
        
        # Initialize random noise for X and Y coordinates of the walk
        walkNoiseX = torch.randn(shape, dtype=self.weightType, device=self.device)
        walkNoiseY = torch.randn(shape, dtype=self.weightType, device=self.device)
        
        # Apply the easing values to the scales (mapping easing values to noise)
        easing_values = self.getEasedBeats(noteType=noteType).to(self.device)
        
        # Compute cumulative sum of easing values to get the angle
        # Adjust the scaling factor to control speed of rotation
        scaling_factor = distance * 2 * math.pi  # Multiply by 2π for full rotations
        radii = easing_values * scaling_factor + spiral_rate * torch.arange(len(easing_values)).to(self.device)
        angles = torch.cumsum(easing_values, dim=0) * scaling_factor
        
        # Compute cos and sin of the angles
        walkScaleX = radii * torch.cos(angles)
        walkScaleY = radii * torch.sin(angles)
        
        # Expand walkScaleX and walkScaleY to match dimensions for multiplication
        walkScaleX = walkScaleX.view(-1, 1, 1, 1)
        walkScaleY = walkScaleY.view(-1, 1, 1, 1)
        
        # Generate the noise tensors based on the interpolated scales
        noiseX = walkScaleX * walkNoiseX
        noiseY = walkScaleY * walkNoiseY
        
        # Add the noise contributions for X and Y to create the latent walk
        latents = noiseX + noiseY  # Shape: (steps, 4, 64, 64)
        return latents

    def getFPS(self):
        return round(self.sr / self.hop_length)
    
    def slerp(self, embed1, embed2, alpha):
        # Normalize embeddings
        embed1_norm = embed1 / torch.norm(embed1, dim=-1, keepdim=True)
        embed2_norm = embed2 / torch.norm(embed2, dim=-1, keepdim=True)

        # Compute the cosine of the angle between embeddings
        dot_product = torch.sum(embed1_norm * embed2_norm, dim=-1, keepdim=True)
        omega = torch.acos(torch.clamp(dot_product, -1.0, 1.0))

        sin_omega = torch.sin(omega)
        if torch.any(sin_omega == 0):
            # Avoid division by zero
            return (1.0 - alpha) * embed1 + alpha * embed2

        # Compute the slerp interpolation
        interp_embed = (
            torch.sin((1.0 - alpha) * omega) / sin_omega * embed1 +
            torch.sin(alpha * omega) / sin_omega * embed2
        )
        return interp_embed
    
    def getPromptEmbedsOnsetFocus(self, basePrompt, targetPromptChromaScale, method="slerp", alpha=0.5, sigma=2):
        chroma = self.chroma_cq.T  # shape: (numFrames, 12)
        numFrames = chroma.shape[0]

        # Onset frames and dominant chromas
        onset_frames = self.onset_bt
        dominant_chromas = self.top_chromaOnset[self.onset_bt]

        # Initialize alphas and dominant chroma per frame
        alphas = np.zeros(numFrames)
        dominant_chroma_per_frame = np.full(numFrames, -1, dtype=int)

        # For each onset interval
        for i in range(len(onset_frames) - 1):
            start_frame = onset_frames[i]
            end_frame = onset_frames[i + 1]
            dominant_chroma = dominant_chromas[i]
            
            if start_frame == end_frame:
                end_frame+=1

            # Extract chroma magnitudes of the dominant chroma
            chroma_magnitudes = chroma[start_frame:end_frame, dominant_chroma]

            # Normalize chroma magnitudes to [0, alpha_max] so that the chroma can experience full visual effect
            min_val = chroma_magnitudes.min()
            max_val = chroma_magnitudes.max()
            if max_val - min_val > 0:
                normalized_magnitudes = (chroma_magnitudes - min_val) / (max_val - min_val)
            else:
                normalized_magnitudes = np.zeros_like(chroma_magnitudes)

            alphas_interval = normalized_magnitudes * alpha
            alphas[start_frame:end_frame] = alphas_interval

            # Assign dominant chroma to frames
            dominant_chroma_per_frame[start_frame:end_frame] = dominant_chroma

        # Handle frames after the last onset
        start_frame = onset_frames[-1]
        dominant_chroma = dominant_chromas[-1]
        end_frame = numFrames

        chroma_magnitudes = chroma[start_frame:end_frame, dominant_chroma]
        min_val = chroma_magnitudes.min()
        max_val = chroma_magnitudes.max()
        if max_val - min_val > 0:
            normalized_magnitudes = (chroma_magnitudes - min_val) / (max_val - min_val)
        else:
            normalized_magnitudes = np.zeros_like(chroma_magnitudes)

        alphas_interval = normalized_magnitudes * alpha
        alphas[start_frame:end_frame] = alphas_interval
        dominant_chroma_per_frame[start_frame:end_frame] = dominant_chroma

        # Apply temporal smoothing to alphas (optional)
        alphas = gaussian_filter1d(alphas, sigma=sigma)

        # Get baseEmbeds and targetEmbeds
        baseInput = self.tokenizer(
            basePrompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
        ).input_ids
        baseEmbeds = self.textEncoder(baseInput)[0].squeeze(0)  # shape: (seq_len, hidden_size)

        targetInputs = self.tokenizer(
            targetPromptChromaScale,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
        ).input_ids
        targetEmbeds = self.textEncoder(targetInputs)[0]  # shape: (12, seq_len, hidden_size)

        # Initialize interpolatedEmbedsAll
        interpolatedEmbedsAll = []

        # For each frame
        for frame in range(numFrames):
            alpha = alphas[frame]
            dominant_chroma = dominant_chroma_per_frame[frame]

            # Use weighted sum of chroma embeddings if desired (optional)
            # For now, we focus on the dominant chroma
            target_embed = targetEmbeds[dominant_chroma]  # shape: (seq_len, hidden_size)

            if method == "linear":
                interpolatedEmbed = (1 - alpha) * baseEmbeds + alpha * target_embed
            elif method == "slerp":
                interpolatedEmbed = self.slerp(baseEmbeds, target_embed, alpha)
            else:
                raise ValueError(f"Unknown interpolation method: {method}")

            interpolatedEmbedsAll.append(interpolatedEmbed.unsqueeze(0))  # shape: (1, seq_len, hidden_size)

        interpolatedEmbeds = torch.stack(interpolatedEmbedsAll)  # shape: (numFrames, 1, seq_len, hidden_size)

        return interpolatedEmbeds
        
        
    def getPromptEmbedsCum(self, basePrompt, targetPromptChromaScale, alpha=0.5, sigma_time=2, sigma_chroma=1):
        """
        Generates interpolated embeddings with multivariate smoothing applied to alpha values.

        Parameters:
        - basePrompt (str): The base prompt text.
        - targetPromptChromaScale (str): The target prompt with chroma scaling.
        - alpha (float): The scaling factor for alpha values.
        - sigma_time (float): Standard deviation for Gaussian kernel in the time dimension.
        - sigma_chroma (float): Standard deviation for Gaussian kernel in the chroma dimension.

        Returns:
        - interpolatedEmbeds (torch.Tensor): The interpolated embeddings tensor.
        """
        # Transpose chroma to shape: (numFrames, 12)
        chroma = self.chroma_cq.T  # shape: (numFrames, 12)
        numFrames = chroma.shape[0]
        number_of_chromas = self.number_of_chromas  # Retrieve the number of chromas to consider

        # Onset frames and top N chromas at onsets
        onset_frames = self.onset_bt
        top_chromas_at_onsets = self.top_chromas[:, self.onset_bt]  # shape: (number_of_chromas, len(onset_bt))

        # Initialize alphas and chroma indices per frame
        alphas = np.zeros((numFrames, number_of_chromas))
        chromas_per_frame = np.full((numFrames, number_of_chromas), -1, dtype=int)

        # For each onset interval
        for i in range(len(onset_frames) - 1):
            start_frame = onset_frames[i]
            end_frame = onset_frames[i + 1]
            chroma_indices = top_chromas_at_onsets[:, i]  # shape: (number_of_chromas,)

            if start_frame == end_frame:
                end_frame += 1

            # Extract chroma magnitudes of the selected chromas in this interval
            chroma_magnitudes = chroma[start_frame:end_frame, chroma_indices]  # shape: (interval_length, number_of_chromas)

            # Normalize chroma magnitudes per frame to sum to alpha
            magnitudes_sum = chroma_magnitudes.sum(axis=1, keepdims=True) + 1e-8  # Avoid division by zero
            alpha_values = (chroma_magnitudes / magnitudes_sum) * alpha  # shape: (interval_length, number_of_chromas)

            # Assign alpha_values and chroma indices to frames
            alphas[start_frame:end_frame, :] = alpha_values
            chromas_per_frame[start_frame:end_frame, :] = chroma_indices.reshape(1, number_of_chromas)

        # Handle frames after the last onset
        start_frame = onset_frames[-1]
        chroma_indices = top_chromas_at_onsets[:, -1]  # shape: (number_of_chromas,)
        end_frame = numFrames

        chroma_magnitudes = chroma[start_frame:end_frame, chroma_indices]  # shape: (interval_length, number_of_chromas)
        magnitudes_sum = chroma_magnitudes.sum(axis=1, keepdims=True) + 1e-8
        alpha_values = (chroma_magnitudes / magnitudes_sum) * alpha

        alphas[start_frame:end_frame, :] = alpha_values
        chromas_per_frame[start_frame:end_frame, :] = chroma_indices.reshape(1, number_of_chromas)

        # **Multivariate Temporal and Chroma Smoothing**
        # Apply a 2D Gaussian filter to smooth across time and chroma dimensions
        # The Gaussian filter requires specifying the sigma for each axis
        # Axis 0: Time (frames), Axis 1: Chroma
        alphas_smoothed = gaussian_filter(alphas, sigma=(sigma_time, sigma_chroma), mode='reflect')

        # **Re-normalize Alphas Per Frame to Ensure the Sum Equals alpha**
        magnitudes_sum = alphas_smoothed.sum(axis=1, keepdims=True) + 1e-8  # Avoid division by zero
        alphas_normalized = (alphas_smoothed / magnitudes_sum) * alpha  # shape: (numFrames, number_of_chromas)

        # **Update the Alphas Array with the Smoothed and Normalized Alphas**
        alphas = alphas_normalized

        # **Proceed with Embedding Interpolation as Before**

        # Get baseEmbeds and targetEmbeds
        baseInput = self.tokenizer(
            basePrompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
        ).input_ids
        baseEmbeds = self.textEncoder(baseInput)[0].squeeze(0)  # shape: (seq_len, hidden_size)

        targetInputs = self.tokenizer(
            targetPromptChromaScale,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
        ).input_ids
        targetEmbeds = self.textEncoder(targetInputs)[0]  # shape: (12, seq_len, hidden_size)

        # Initialize interpolatedEmbedsAll
        interpolatedEmbedsAll = []

        # For each frame
        for frame in range(numFrames):
            alpha_values = alphas[frame, :]  # shape: (number_of_chromas,)
            total_alpha = alpha_values.sum()
            if total_alpha > alpha:
                alpha_values = (alpha_values / total_alpha) * alpha
                total_alpha = alpha

            base_alpha = 1.0 - total_alpha
            chroma_indices = chromas_per_frame[frame, :]  # shape: (number_of_chromas,)

            # Start with baseEmbeds multiplied by base_alpha
            interpolatedEmbed = base_alpha * baseEmbeds

            # Add contributions from each target chroma
            for n in range(number_of_chromas):
                target_embed = targetEmbeds[chroma_indices[n]]  # shape: (seq_len, hidden_size)
                interpolatedEmbed += alpha_values[n] * target_embed

            interpolatedEmbedsAll.append(interpolatedEmbed.unsqueeze(0))  # shape: (1, seq_len, hidden_size)

        interpolatedEmbeds = torch.stack(interpolatedEmbedsAll)  # shape: (numFrames, seq_len, hidden_size)

        return interpolatedEmbeds
    # def getPromptEmbedsCum(self, basePrompt, targetPromptChromaScale, alpha=0.5, sigma_time=2, sigma_chroma=1):
    #     chroma = self.chroma_cq.T  # shape: (numFrames, 12)
    #     numFrames = chroma.shape[0]
    #     number_of_chromas = self.number_of_chromas  # Retrieve the number of chromas to consider

    #     # Onset frames and top N chromas at onsets
    #     onset_frames = self.onset_bt
    #     top_chromas_at_onsets = self.top_chromas[:, self.onset_bt]  # shape: (number_of_chromas, len(onset_bt))

    #     # Initialize alphas and chroma indices per frame
    #     alphas = np.zeros((numFrames, number_of_chromas))
    #     chromas_per_frame = np.full((numFrames, number_of_chromas), -1, dtype=int)

    #     # For each onset interval
    #     for i in range(len(onset_frames) - 1):
    #         start_frame = onset_frames[i]
    #         end_frame = onset_frames[i + 1]
    #         chroma_indices = top_chromas_at_onsets[:, i]  # shape: (number_of_chromas,)

    #         if start_frame == end_frame:
    #             end_frame += 1

    #         # Extract chroma magnitudes of the selected chromas in this interval
    #         chroma_magnitudes = chroma[start_frame:end_frame, chroma_indices]  # shape: (interval_length, number_of_chromas)

    #         # Normalize chroma magnitudes per frame to sum to alpha
    #         magnitudes_sum = chroma_magnitudes.sum(axis=1, keepdims=True) + 1e-8  # Avoid division by zero
    #         alpha_values = (chroma_magnitudes / magnitudes_sum) * alpha  # shape: (interval_length, number_of_chromas)

    #         # Assign alpha_values and chroma indices to frames
    #         alphas[start_frame:end_frame, :] = alpha_values
    #         chromas_per_frame[start_frame:end_frame, :] = chroma_indices.reshape(1, number_of_chromas)

    #     # Handle frames after the last onset
    #     start_frame = onset_frames[-1]
    #     chroma_indices = top_chromas_at_onsets[:, -1]  # shape: (number_of_chromas,)
    #     end_frame = numFrames

    #     chroma_magnitudes = chroma[start_frame:end_frame, chroma_indices]  # shape: (interval_length, number_of_chromas)
    #     magnitudes_sum = chroma_magnitudes.sum(axis=1, keepdims=True) + 1e-8
    #     alpha_values = (chroma_magnitudes / magnitudes_sum) * alpha

    #     alphas[start_frame:end_frame, :] = alpha_values
    #     chromas_per_frame[start_frame:end_frame, :] = chroma_indices.reshape(1, number_of_chromas)

    #     # Apply temporal smoothing to alphas (optional)
    #     for n in range(number_of_chromas):
    #         alphas[:, n] = gaussian_filter1d(alphas[:, n], sigma=sigma)

    #     # Get baseEmbeds and targetEmbeds
    #     baseInput = self.tokenizer(
    #         basePrompt,
    #         return_tensors="pt",
    #         padding="max_length",
    #         truncation=True,
    #         max_length=77,
    #     ).input_ids
    #     baseEmbeds = self.textEncoder(baseInput)[0].squeeze(0)  # shape: (seq_len, hidden_size)

    #     targetInputs = self.tokenizer(
    #         targetPromptChromaScale,
    #         return_tensors="pt",
    #         padding="max_length",
    #         truncation=True,
    #         max_length=77,
    #     ).input_ids
    #     targetEmbeds = self.textEncoder(targetInputs)[0]  # shape: (12, seq_len, hidden_size)

    #     # Initialize interpolatedEmbedsAll
    #     interpolatedEmbedsAll = []

    #     # For each frame
    #     for frame in range(numFrames):
    #         alpha_values = alphas[frame, :]  # shape: (number_of_chromas,)
    #         total_alpha = alpha_values.sum()
    #         if total_alpha > alpha:
    #             alpha_values = (alpha_values / total_alpha) * alpha
    #             total_alpha = alpha

    #         base_alpha = 1.0 - total_alpha
    #         chroma_indices = chromas_per_frame[frame, :]  # shape: (number_of_chromas,)

    #         # Start with baseEmbeds multiplied by base_alpha
    #         interpolatedEmbed = base_alpha * baseEmbeds

    #         # Add contributions from each target chroma
    #         for n in range(number_of_chromas):
    #             target_embed = targetEmbeds[chroma_indices[n]]  # shape: (seq_len, hidden_size)
    #             interpolatedEmbed += alpha_values[n] * target_embed

    #         interpolatedEmbedsAll.append(interpolatedEmbed.unsqueeze(0))  # shape: (1, seq_len, hidden_size)

    #     interpolatedEmbeds = torch.stack(interpolatedEmbedsAll)  # shape: (numFrames, seq_len, hidden_size)

    #     return interpolatedEmbeds
    
    def getVisuals(self, latents, promptEmbeds, num_inference_steps=1, batch_size=1):
        #self.pipe.vae = self.vae
        self.pipe.to(device=self.device, dtype=self.weightType)
        
        latents.to(self.device)
        promptEmbeds.to(self.device)
        
        allFrames=[]
        
        num_frames = self.steps
        
        for i in tqdm.tqdm(range(0, num_frames, batch_size)):
            frames = self.pipe(prompt_embeds=promptEmbeds[i],
                       guidance_scale=0,
                       num_inference_steps=num_inference_steps,
                       latents=latents[i:i+batch_size],
                       output_type="pil").images
            allFrames.extend(frames)
        return allFrames
        
        