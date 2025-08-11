import os, sys
import argparse, glob, cv2

import torch
from torchvision import transforms
from customdiffusers.models.controlnet import AlignNet
from customdiffusers.pipelines.controlnet.pipeline_controlnet_sd_xl import ALignSDXLPipeline

from PIL import Image

from align_utils import simple_neg, style_prompts
import dlib
import copy
from tqdm import tqdm

#cache_dir = os.path.abspath('./cachedir')
#os.environ["HF_HOME"] =cache_dir
#os.environ['TRANSFORMERS_CACHE'] =cache_dir

base_model_path = "SG161222/RealVisXL_V3.0"
device = "cuda"
eam_path = "./EAM_ckpt"


img_idx=0
import random
import numpy as np
def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_styles', type=int,default=3) 
    args = parser.parse_args()
    
    EAM = AlignNet.from_pretrained(eam_path, use_safetensors=True, torch_dtype=torch.float16,cache_dir=cache_dir).to(device)
    EAM.eval()
    pipe = ALignSDXLPipeline.from_pretrained(
        base_model_path,
        controlnet=EAM,
        use_safetensors=True,
        torch_dtype=torch.float16,
        add_watermarker=False,
        cache_dir=cache_dir
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    




    
    vae_32 = copy.deepcopy(pipe.vae).float()
    all_data_path = 'face_fixed/params_10k.pth'
    all_data = torch.load(all_data_path)
    for exidx,example in enumerate(tqdm(all_data)):
            
            
        # Process image for landmark detection
        image_path = example['name']
        image = Image.open(f'face_fixed/{image_path}')
        
        
        emotion = example['flame'][0,-100:-50].to(device)
        pose = torch.tensor(example['rotation']).to(device)
        
        image_tensor = pipe.image_processor.preprocess(image).to(device)
        source_latent = vae_32.encode(image_tensor).latent_dist.sample() 
        source_latent*= pipe.vae.config.scaling_factor

        
        #pitch, yaw, roll = pose_model.predict(img_cv)
        validation_lmk = torch.tensor(example['lmk']).to(device).unsqueeze(0)
        source_image = image.resize((512, 512))
        all_keys = list(style_prompts.keys())[:args.num_styles]
        
        for styleint, style_key in enumerate(all_keys):
            
            style_prompt = style_prompts[style_key]
            os.makedirs(f'output_test/{style_key}',exist_ok=True)
            image= pipe( prompt=style_prompt,negative_prompt=simple_neg,num_samples=1,latents=source_latent, controlnet_conditioning_scale=0.7,lmk=validation_lmk, pose=pose,exp=emotion,num_inference_steps=25,skip_steps_injection=6).images
            
            image[0].save(f'output_test/{style_key}/{img_idx:04}.png')    
        img_idx+=1



    
