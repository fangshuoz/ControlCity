# ControlCity: A Multimodal Diffusion Model Based Approach for Accurate Geospatial Data Generation and Urban Morphology Analysis

[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-green)](https://huggingface.co/fangshuoz/ControlCity)


Fangshuo Zhou,
Huaxia Li,
[Liuchang Xu](https://www.researchgate.net/profile/Liuchang-Xu)<sup>*</sup>

<sup>*</sup>corresponding authors

## 📢 News
2024-09-24: We uploaded sample data for visitors to perform inference with the model. <br>
2024-09-19: We uploaded the model to HuggingFace <a href="https://huggingface.co/fangshuoz/ControlCity"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Model-green" height="16"></a>. <br>
2024-09-13: ControlCity official github repository is officially created.

## 📦 Repository
Clone the repository (requires git):
```bash
git clone https://github.com/fangshuoz/ControlCity.git

pip install -r requirements.txt
```

## 🚀 Quickstart

```python
from PIL import Image
from DiffusionOSM.diffusionosm import (
    OSMControlNetModel,
    DiffusionOSMControlnetPipeline,
    metadata_normalize,
    convert_binary,
)
from diffusers import UniPCMultistepScheduler
import torch

# load pipeline
controlnet = OSMControlNetModel.from_pretrained(
    trained_controlnet_model_path,
    torch_dtype=torch.float16, use_safetensors=True,
    low_cpu_mem_usage=False, device_map=None
)
pipe = DiffusionOSMControlnetPipeline.from_pretrained(
    sdxl_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    use_safetensors=True,
)
pipe.load_lora_weights(
    trained_lora_model_path,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to('cuda:1')

# load condition(text, metadata, cond_image, etc.)
metadata = [-122.3382568359375, 47.61727258456622]
prompt = "A black and white map of city buildings, Located in Seattle, Mostly urban area with numerous buildings, parking lots, ..."
image_road = Image.open('road/15/Seattle/5248_11443.png').convert("RGB")
image_landuse = Image.open('landuse/15/Seattle/5248_11443.png').convert("RGB")

metadata = metadata_normalize(metadata).tolist()

# inference
image = pipe(
    prompt=prompt,
    metadata=metadata,
    negative_prompt="Low quality.",
    image_road=image_road,
    image_landuse=image_landuse,
    guidance_scale=5.0,
    num_inference_steps=25,
    generator=torch.manual_seed(42)
).images[0]

image_bin = convert_binary(image, thr=60, mode="RGB", image_landuse=image_landuse)[0]
```
