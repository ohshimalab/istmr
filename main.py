import os
from pathlib import Path

import pandas as pd
import typer
from typer import Typer
from typing_extensions import Annotated

from istmr.utils import download_civitai_model, download_image_from_unsplash, generate_image_from_pair

app = Typer()


@app.command()
def download_content_images(
    content_image_list_excel: Annotated[
        Path, typer.Option(help="Content Images metatdata excel")
    ] = "./data/content-images.xlsx",
    output_dir: Annotated[Path, typer.Option(help="Directory to save downloaded images")] = "./data/content-images/",
):
    """
    Download images from a list in an Excel file and save them to the specified directory.

    Args:
        content_image_list_excel (str): Path to the Excel file containing image URLs.
        output_dir (str): Directory where images will be saved.
    """

    df = pd.read_excel(content_image_list_excel)
    if "image_url" not in df.columns:
        raise ValueError("Excel file must contain a column named 'image_url'.")

    os.makedirs(output_dir, exist_ok=True)

    for _, row in df.iterrows():
        image_url = row["image_url"]
        image_id = row["image_id"]
        save_path = os.path.join(output_dir, f"{image_id}.jpg")
        try:
            download_image_from_unsplash(image_url, save_path)
            print(f"Downloaded image {image_id} from {image_url} to {save_path}")
        except Exception as e:
            print(f"Failed to download image {image_id} from {image_url}: {e}")


@app.command()
def download_civitai_models(
    model_list_excel: Annotated[Path, typer.Option(help="Civitai Models metadata excel")] = "./data/models.xlsx",
    output_dir: Annotated[Path, typer.Option(help="Directory to save downloaded models")] = "./data/models/",
    token: Annotated[str, typer.Option(help="Civitai API token")] = "",
):
    """
    Download models from Civitai based on a list in an Excel file and save them to the specified directory.

    Args:
        model_list_excel (str): Path to the Excel file containing model IDs.
        output_dir (str): Directory where models will be saved.
        token (str): Civitai API token for authentication.
    """

    df = pd.read_excel(model_list_excel)
    if "model_id" not in df.columns:
        raise ValueError("Excel file must contain a column named 'model_id'.")

    os.makedirs(output_dir, exist_ok=True)

    for _, row in df.iterrows():
        model_id, version_id, model_version_id, url = (
            row["model_id"],
            row["version_id"],
            row["model_version_id"],
            row["url"],
        )
        save_path = os.path.join(output_dir, f"{model_version_id}.safetensors")
        try:
            download_civitai_model(version_id, save_path, token)
            print(f"Downloaded model {model_id} to {save_path}")
        except Exception as e:
            print(f"Failed to download model {model_id}: {e}")


@app.command()
def generate_pair(
    content_image_dir: Annotated[
        Path, typer.Option(help="Directory containing content images")
    ] = "./data/content-images/",
    content_image_id: Annotated[str, typer.Option(help="Content image ID to use for generation")] = "1",
    model_dir: Annotated[Path, typer.Option(help="Directory containing models")] = "./data/models/",
    model_version_id: Annotated[str, typer.Option(help="Model version ID to use for generation")] = "4844_5571",
    model_trigger_word: Annotated[str, typer.Option(help="Trigger word for the model")] = "",
    seed: Annotated[int, typer.Option(help="Random seed for generation")] = 0,
    output_dir: Annotated[Path, typer.Option(help="Directory to save generated images")] = "./data/generated-images/",
):
    save_path = (
        output_dir / str(content_image_id) / model_version_id / f"{model_version_id}_{content_image_id}_{seed}.jpg"
    )
    generated_image = generate_image_from_pair(
        content_image_dir=content_image_dir,
        content_image_id=content_image_id,
        model_dir=model_dir,
        model_version_id=model_version_id,
        seed=seed,
        model_trigger_word=model_trigger_word,
    )
    generated_image.save(save_path)


if __name__ == "__main__":
    app()
