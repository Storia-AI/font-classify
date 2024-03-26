import argparse
import albumentations as A
import csv
import huggingface_hub
import numpy as np
import onnxruntime as ort
import os
import yaml

from PIL import Image
from train import CutMax, ResizeWithPad


CONFIG_PATH = huggingface_hub.hf_hub_download(
    repo_id="storia/font-classify-onnx", filename="model_config.yaml"
)
MODEL_PATH = huggingface_hub.hf_hub_download(
    repo_id="storia/font-classify-onnx", filename="model.onnx"
)
MAPPING_PATH = "google_fonts_mapping.tsv"


def parse_args():
    # Create an argument parser
    parser = argparse.ArgumentParser(
        description="Inference with pretrained model from Storia"
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default="sample_data/output/Pacifico-Regular",
        help="Path to images to run inference on",
    )
    args = parser.parse_args()
    return args


def softmax(x):
    """Computes softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # axis=0 for 2d array case


def main(args):
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    input_size = config["size"]

    google_font_mapping = {}
    with open(MAPPING_PATH, "r") as f:
        tsv_file = csv.reader(f, delimiter="\t")
        for i, row in enumerate(tsv_file):
            if i > 0:
                filename, font_name, version = row
                google_font_mapping[filename] = (font_name, version)

    session = ort.InferenceSession(MODEL_PATH)

    transform = A.Compose(
        [
            A.Lambda(image=CutMax(1024)),
            A.Lambda(image=ResizeWithPad((input_size, input_size))),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    for image_file in os.listdir(args.data_folder):
        image_path = os.path.join(args.data_folder, image_file)
        image = np.array(Image.open(image_path))
        image = transform(image=image)["image"]
        # Move the channel dimension to the front.
        image = np.transpose(image, (2, 0, 1))
        # Add a dummy batch dimension.
        image = np.expand_dims(image, 0)

        logits = session.run(None, {"input": image})[0][0]
        probs = softmax(logits)
        predicted = config["classnames"][probs.argmax(0)]
        print(image_file, *google_font_mapping.get(predicted))


if __name__ == "__main__":
    args = parse_args()

    main(args)
