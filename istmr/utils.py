import random
from pathlib import Path

import numpy as np
import requests as python_requests
import torch
from bs4 import BeautifulSoup
from curl_cffi import requests
from diffusers import DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline
from PIL import Image


def download_image_from_unsplash(url, save_path):
    """
    Download an image from Unsplash and save it to the specified path.

    Args:
        url (str): The URL of the image to download.
        save_path (str): The path where the image will be saved.
    """
    image_page_response = requests.get(url, impersonate="chrome101")
    if image_page_response.status_code != 200:
        raise Exception(f"Failed to retrieve the image page. Status code: {image_page_response.status_code}")
    soup = BeautifulSoup(image_page_response.content, "html.parser")
    download_link_elem = soup.select_one(".dOeZz a")
    download_url = download_link_elem["href"]
    # download the image
    image_response = requests.get(download_url, impersonate="chrome101")
    if image_response.status_code != 200:
        raise Exception(f"Failed to download the image. Status code: {image_response.status_code}")
    with open(save_path, "wb") as file:
        file.write(image_response.content)


def download_civitai_model(model_id, save_path, token):
    """
    Download a model from Civitai and save it to the specified path.

    Args:
        model_id (str): The ID of the model to download.
        save_path (str): The path where the model will be saved.
    """
    url = f"https://civitai.com/api/download/models/{model_id}?token={token}"
    download_response = python_requests.get(url)
    with open(save_path, "wb") as file:
        file.write(download_response.content)


def get_model_pipeline(model_dir: Path, model_version_id: str):
    # Load the pre-trained Stable Diffusion pipeline
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "shibal1/anything-v4.5-clone",
        torch_dtype=torch.float16,
    )

    # DPM++ 2M Karras
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True  # <-- turn on Karras sigmas
    )

    # Load LoRA
    model_path = model_dir / f"{model_version_id}.safetensors"
    pipe.load_lora_weights(str(model_path))

    # Negative Embedding
    pipe.load_textual_inversion(
        "gsdf/Counterfeit-V3.0",
        weight_name="embedding/EasyNegativeV2.safetensors",
        token="EasyNegativeV2",
    )
    pipe.load_textual_inversion(
        "Eugeoter/badhandv4",
        weight_name="badhandv4.safetensors",
        token="badhandv4",
    )

    pipe.safety_checker = lambda images, **kwargs: (images, None)
    return pipe


def get_fixed_generator(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    generator = torch.Generator(device="cuda").manual_seed(seed)
    return generator


def resize_image(image, target_long_edge=1024):
    """
    Resize the given PIL Image so that its longer edge is target_long_edge pixels,
    preserving the aspect ratio.

    Parameters:
        image (PIL.Image): The image to resize.
        target_long_edge (int): The desired size for the longer edge (default is 1024).

    Returns:
        PIL.Image: The resized image.
    """
    original_width, original_height = image.size

    if original_width >= original_height:
        # Landscape (or square): width is longer edge
        new_width = target_long_edge
        new_height = int((target_long_edge / original_width) * original_height)
    else:
        # Portrait: height is longer edge
        new_height = target_long_edge
        new_width = int((target_long_edge / original_height) * original_width)

    # Resize using the LANCZOS filter for high-quality downsizing
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    return resized_image


def load_content_image(content_image_dir, content_image_id, res: int = 1024):
    image_path = content_image_dir / f"{content_image_id}.jpg"
    content_image = Image.open(image_path).convert("RGB")

    content_image = resize_image(content_image, res)
    return content_image


def generate_image(
    content_image: Image.Image,
    pipline: StableDiffusionImg2ImgPipeline,
    strength: float = 0.6,
    trigger_word: str = "",
    negative_prompt: str = "",
    seed: int = 0,
):
    prompt = f"((masterpiece, ultra quality)), {trigger_word}"
    negative_prompt = f"EasyNegativeV2, badhandv4, nsfw, {negative_prompt}"
    generator = get_fixed_generator(seed)
    result = pipline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=content_image,
        strength=strength,
        clip=2,
        generator=generator,
    )
    return result.images[0]


def generate_image_from_pair(
    model_dir: Path,
    model_version_id: str,
    model_trigger_word: str,
    content_image_dir: Path,
    content_image_id: str,
    seed: int = 0,
    use_cuda: bool = True,
):
    content_image = load_content_image(content_image_dir, content_image_id)
    pipline = get_model_pipeline(model_dir, model_version_id)
    if use_cuda:
        pipline.to("cuda")
    else:
        pipline.to("cpu")
    result_image = generate_image(
        content_image=content_image,
        pipline=pipline,
        trigger_word=model_trigger_word,
        seed=seed,
    )
    return result_image
