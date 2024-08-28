import torch
#from diffusers import StableDiffusionPipeline
from diffusers import FluxPipeline
import librosa
from PIL import Image
import numpy as np
import moviepy.editor as mpy
import math

from utils import LatentInitCircular



torch.manual_seed(69420)
class NoiseVisualizer:
    def __init__(self, device="mps", weightType=torch.float16):
        self.device = device
        self.weightType = weightType
        #self.pipe = StableDiffusionPipeline.from_pretrained("IDKiro/sdxs-512-dreamshaper", torch_dtype=weightType)
        self.pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        self.textEncoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer
        
    def loadSong(self,file,hop_length=512):
        y, sr = librosa.load(file) # 3 min 52 sec
        self.hop_length=hop_length
        self.sr = sr
        self.y = y
        
        #create spectrogram
        spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000, 
                                              hop_length=self.hop_length)

        #get mean power at each time point
        specm=np.mean(spec,axis=0)

        #compute power gradient across time points
        gradm=np.gradient(specm)

        #set max to 1
        self.gradm=gradm/np.max(gradm)

        #set negative gradient time points to zero 
        #self.gradm = gradm.clip(min=0)
            
        #normalize mean power between 0-1
        self.specm = (specm-np.min(specm))/np.ptp(specm)
        
        self.chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=self.hop_length)
        
        self.chromaGrad = np.gradient(self.chroma)
        
    def getSpecCircleLatents(self,distance=2):
        noiseCircle = LatentInitCircular(steps = self.specm.shape[0],distance=distance)
        specmScaled = (self.specm * (self.specm.shape[0] - 1)).astype('int32')
        latents = noiseCircle[specmScaled]
        latents = latents.squeeze(1)
        return latents
    
    def getFPS(self):
        return self.sr / self.hop_length 
    
    def getPromptEmbeds(self, basePrompt, targetPromptChromaScale, alpha=0.5):
        #example: basePrompt= "Demon Dancing, demon dancing cheerfully, blue background, jazzy, bells, creepy and morose, hardcore graphics"
        # targetPromptChromaScale = ["Black", "Dark Blue", "Purple", "Magenta", "Red", "Orange", "Yellow", "Sea green", "Green", "Teal", "Light Blue", "White"]
        
        chroma = self.chroma.T #shape-> n,12
        alphas = np.full(chroma.shape,alpha)
        chroma = chroma * alphas
        numFrames = chroma.shape[0]
        
        baseInput = self.tokenizer(basePrompt, return_tensors="pt", padding="max_length", truncation = True, max_length = 16).input_ids
        targetInput = self.tokenizer(targetPromptChromaScale, return_tensors="pt", padding="max_length", truncation = True, max_length = 16).input_ids
        
        baseEmbeds = self.textEncoder(baseInput)[0] # shape-> 1,3,768
        targetEmbeds = self.textEncoder(targetInput)[0] #shape-> 12,3,768
        
        #baseEmbeds = baseEmbeds.unsqueeze(0).expand(numFrames, -1, -1 , -1) #shape -> n, 1, 3, 768
        targetEmbeds = targetEmbeds.unsqueeze(0).expand(numFrames, -1, -1, -1) #shape -> n, 12, 3, 768
        
        interpolatedEmbedsAll = []
        for frameNum in range(numFrames):
            interpolatedEmbed = baseEmbeds.clone()
            for chromaKey in range(12):
                interpolatedEmbed = (1 - chroma[frameNum,chromaKey]) * interpolatedEmbed + \
                chroma[frameNum,chromaKey] * targetEmbeds[frameNum,chromaKey]
                
            interpolatedEmbedsAll.append(interpolatedEmbed)
            
        interpolatedEmbeds = torch.stack(interpolatedEmbedsAll)
        return interpolatedEmbeds
        
        
      
    def getVisuals(self, latents, promptEmbeds, num_inference_steps=1,guidance_scale = 0.3, batch_size=1):
        #self.pipe.vae = self.vae
        self.pipe.to(device=self.device, dtype=self.weightType)
        
        latents.to(self.device)
        promptEmbeds.to(self.device)
        
        allFrames=[]
        
        num_frames = latents.shape[0]
        
        for i in range(0, num_frames, batch_size):
            frames = self.pipe(prompt_embeds = promptEmbeds[i],
                                guidance_scale = guidance_scale,
                                num_inference_steps = num_inference_steps,
                                latents=latents[i:i+batch_size],
                                output_type="pil").images
            allFrames.extend(frames)
        return allFrames
        
        # frame = self.pipe(prompt=prompt,
        #                       guidance_scale=0.0,
        #                       num_inference_steps=1,
        #                       output_type="pil").images[0]
        # return frame
        