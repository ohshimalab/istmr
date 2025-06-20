import os
import pickle
from pathlib import Path

import pandas as pd
import typer
from rich import print
from typer import Typer
from typing_extensions import Annotated

from istmr.utils import (
    create_data_module,
    download_civitai_model,
    download_image_from_unsplash,
    generate_image_from_pair,
    train_model,
)

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
    save_path.parent.mkdir(parents=True, exist_ok=True)
    generated_image = generate_image_from_pair(
        content_image_dir=content_image_dir,
        content_image_id=content_image_id,
        model_dir=model_dir,
        model_version_id=model_version_id,
        seed=seed,
        model_trigger_word=model_trigger_word,
    )
    generated_image.save(save_path)


@app.command()
def generate_images(
    content_image_dir: Annotated[
        Path, typer.Option(help="Directory containing content images")
    ] = "./data/content-images/",
    model_dir: Annotated[Path, typer.Option(help="Directory containing models")] = "./data/models/",
    output_dir: Annotated[Path, typer.Option(help="Directory to save generated images")] = "./data/generated-images/",
    data_csv: Annotated[
        Path, typer.Option(help="CSV file containing content image IDs and model version IDs")
    ] = "./data/data.csv",
    model_excel: Annotated[Path, typer.Option(help="Excel file containing model metadata")] = "./data/models.xlsx",
):
    df_data = pd.read_csv(data_csv)
    df_models = pd.read_excel(model_excel)
    model_trigger_words_dict = df_models.set_index("model_version_id")["trigger_words"].to_dict()
    for _, row in df_data.iterrows():
        model_version_id = row["model_version_id"]
        content_image_id = row["original_image_id"]
        seed = row["seed"]
        model_trigger_word = model_trigger_words_dict.get(model_version_id, "")
        save_path = (
            output_dir / str(content_image_id) / model_version_id / f"{model_version_id}_{content_image_id}_{seed}.jpg"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        generated_image = generate_image_from_pair(
            content_image_dir=content_image_dir,
            content_image_id=content_image_id,
            model_dir=model_dir,
            model_version_id=model_version_id,
            seed=seed,
            model_trigger_word=model_trigger_word,
        )
        generated_image.save(save_path)


@app.command()
def train_and_evaluate_proposed_method(
    generated_image_dir: Annotated[
        Path, typer.Option(help="Directory containing content images")
    ] = "./data/generated-images/",
    alpha: Annotated[float, typer.Option(help="Alpha value for the model")] = 0.5,
    vit_name: Annotated[
        str, typer.Option(help="Name of the ViT model to use (e.g., 'vit_base_patch16_224')")
    ] = "vit_base_patch16_224",
    num_transformer_layers: Annotated[int, typer.Option(help="Number of transformer layers in the model")] = 2,
    num_heads: Annotated[int, typer.Option(help="Number of attention heads in the model")] = 8,
    num_models: Annotated[int, typer.Option(help="Number of models to train on")] = 100,
    batch_size: Annotated[int, typer.Option(help="Batch size for training")] = 16,
    max_epochs: Annotated[int, typer.Option(help="Maximum number of epochs for training")] = 30,
    num_workers: Annotated[int, typer.Option(help="Number of workers for data loading")] = 4,
    data_csv: Annotated[
        Path, typer.Option(help="CSV file containing content image IDs and model version IDs")
    ] = "./data/data.csv",
    model_save_path: Annotated[Path, typer.Option(help="Path to save the trained model")] = "./data/trained_model.pth",
    vit_save_path: Annotated[Path, typer.Option(help="Path to save the ViT model")] = "./data/vit_model.pth",
    lr: Annotated[float, typer.Option(help="Learning rate for training")] = 1e-5,
    seed: Annotated[int, typer.Option(help="Random seed for reproducibility")] = 0,
    test_results_csv: Annotated[Path, typer.Option(help="Path to save test results CSV")] = "./data/test_results.csv",
):
    df_data = pd.read_csv(data_csv)
    data_module = create_data_module(
        root_dir=generated_image_dir,
        df_data=df_data,
        batch_size=batch_size,
        num_workers=num_workers,
        vit_name=vit_name,
    )
    trained_vit, trained_model, df_test_metrics = train_model(
        alpha=alpha,
        vit_name=vit_name,
        num_transformer_layers=num_transformer_layers,
        num_heads=num_heads,
        num_models=num_models,
        data_module=data_module,
        lr=lr,
        max_epochs=max_epochs,
        seed=seed,
    )
    with open(model_save_path, "wb") as f:
        pickle.dump(trained_model, f)
    with open(vit_save_path, "wb") as f:
        pickle.dump(trained_vit, f)
    df_test_metrics.to_csv(test_results_csv, index=False)
    print(f"Model saved to {model_save_path}")
    print(f"ViT model saved to {vit_save_path}")
    print(f"Test metrics saved to {test_results_csv}")


@app.command()
def fix_seed():
    df_data = pd.read_csv("./data/data.csv")

    def new_generated_path(path):
        seed = path.split("_")[-1].split(".")[0]
        new_seed = int(seed) + 1
        return path.replace(f"_{seed}.jpg", f"_{new_seed}.jpg")

    df_data["generated_image_path"] = df_data["generated_image_path"].apply(new_generated_path)
    df_data.to_csv("./data/data.csv", index=False)


if __name__ == "__main__":
    app()
