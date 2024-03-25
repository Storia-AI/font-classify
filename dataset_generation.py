"""Script to generate data for the font classification task.

Sample run:
```
python -m venv my-venv
source my-venv/bin/activate
pip install -r requirements.dataset_generation.txt

python dataset_generation.py 100
```
"""

import colorsys
import cv2
import numpy as np
import os
import sys
import random
import traceback
import wikipedia

from PIL import Image, ImageDraw, ImageFont
from argparse import ArgumentParser
from loguru import logger
from pathlib import Path
from sklearn.cluster import KMeans
from tqdm import tqdm
from typing import Tuple, Optional


logger.remove()
logger.add(sys.stdout, level="INFO")


def get_common_colors(
    img, colors=32, max_points=-1, N=3, colorspace="rgb", select_color="mean"
):
    max_points = int(max_points)
    img = np.array(img, dtype=np.uint8)
    h, w = img.shape[0], img.shape[1]

    img_orig_flat = img.reshape(h * w, 3)

    if colorspace == "bgr":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif colorspace == "hls":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif colorspace == "hsv":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif colorspace == "lab":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    elif colorspace == "rgb":
        img = img.copy()
    else:
        raise Exception("Unknown colorspace")

    img_flat = img.copy().reshape(h * w, 3)

    if max_points > 0 and max_points < img_flat.shape[0]:
        idx = np.random.choice(np.arange(img_flat.shape[0]), max_points, replace=False)
        kmeans = KMeans(n_clusters=colors, n_init="auto", random_state=0).fit(
            img_flat[idx]
        )
        labels = kmeans.predict(img_flat)
    else:
        kmeans = KMeans(n_clusters=colors, n_init="auto", random_state=0).fit(img_flat)
        labels = kmeans.labels_

    unique_labels, counts = np.unique(labels, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]
    most_common_labels = unique_labels[sorted_indices[:N]]

    # loops for cluster center
    colors = []
    for ci in np.unique(most_common_labels):
        if select_color == "mean":
            colors.append(img_orig_flat[labels == ci, :].mean(axis=0))
        elif select_color == "median":
            colors.append(np.median(img_orig_flat[labels == ci, :], axis=0))
        else:
            raise Exception("Unknown select_color")
    return [c.astype(np.uint8) for c in colors]


def load_image(image_path):
    return Image.open(image_path).convert("RGB")


def rgb_to_hls(rgb):
    return colorsys.rgb_to_hls(*[x / 255.0 for x in rgb])


def hls_to_rgb(hls):
    return tuple([int(x * 255) for x in colorsys.hls_to_rgb(*hls)])


def triadic_color_hls(rgb):
    h, l, s = rgb_to_hls(rgb)
    # s = max(0.7, s)
    # FIXME: dirty hack for inverse black to white and back
    # TODO: make some threshold that will define "dark" and "white" colors
    # and inverse brightness for them
    # v, s = s, v
    l = 1.0 - l
    h_triadic1 = (h + 1 / 3) % 1
    h_triadic2 = (h + 2 / 3) % 1
    return hls_to_rgb((h_triadic1, l, s)), hls_to_rgb((h_triadic2, l, s))


def opposite_color_hls(rgb):
    h, l, s = rgb_to_hls(rgb)
    l = 1.0 - l
    h_opposite = (h + 1 / 2) % 1
    return hls_to_rgb((h_opposite, l, s))


def get_random_page_content() -> str:
    page_title = wikipedia.random(1)
    try:
        page_content = wikipedia.page(page_title).summary
    except (wikipedia.DisambiguationError, wikipedia.PageError):
        return get_random_page_content()
    return page_content


def split_string(string, min_length, max_length):
    substrings = []
    start = 0
    length = len(string)

    for i in range(length // max_length):
        substr = string[start : start + max_length]
        start += max_length
        substrings.append(substr)

    if length - start > min_length:
        substrings.append(string[start:])

    return substrings


def create_strings_from_wikipedia(minimum_length, count, lang, max_length=-1):
    """
    Create all string by randomly picking Wikipedia articles and taking sentences from them.
    """
    wikipedia.set_lang(lang)
    sentences = []

    while len(sentences) < count:
        page_content = get_random_page_content()
        processed_content = page_content.replace("\n", " ").split(". ")
        sentence_candidates = [
            s.strip() for s in processed_content if len(s.split()) > minimum_length
        ]

        for candidate in sentence_candidates:
            strings = split_string(candidate, minimum_length, max_length)
            if len(strings) > 0:
                sentences.extend(strings)
        # sentences.extend(sentence_candidates)

    return sentences[0:count]


def create_strings_from_textfile(textfile_path, min_length, max_length, count=-1):
    with open(textfile_path, "r") as f:
        lines = f.readlines()

    sentences = []
    for line in lines:
        if len(line) > min_length:
            strings = split_string(line, min_length, max_length)
            sentences.extend(strings)

        if count > 0 and len(sentences) >= count:
            break

    return sentences[0:count]


class ResizeWithPad:

    def __init__(
        self, new_shape: Tuple[int, int], padding_color: Tuple[int] = (255, 255, 255)
    ) -> None:
        self.new_shape = new_shape
        self.padding_color = padding_color

    def __call__(self, image: np.array, padding_color=None, **kwargs) -> np.array:
        """Maintains aspect ratio and resizes with padding.
        Params:
            image: Image to be resized.
            new_shape: Expected (width, height) of new image.
            padding_color: Tuple in BGR of padding color
        Returns:
            image: Resized image with padding
        """
        if padding_color is None:
            padding_color = self.padding_color
        original_shape = (image.shape[1], image.shape[0])
        ratio = float(max(self.new_shape)) / max(original_shape)
        new_size = tuple([int(x * ratio) for x in original_shape])
        image = cv2.resize(image, new_size)
        delta_w = self.new_shape[0] - new_size[0]
        delta_h = self.new_shape[1] - new_size[1]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color
        )
        return image


class CutMax:
    """Cuts the image to the maximum size"""

    def __init__(self, max_size: int = 1024) -> None:
        self.max_size = max_size

    def __call__(self, image: np.array, **kwargs) -> np.array:
        """Cuts the image to the maximum size"""
        if image.shape[0] > self.max_size:
            image = image[: self.max_size, :, :]
        if image.shape[1] > self.max_size:
            image = image[:, : self.max_size, :]
        return image


class FontGenerator:
    """
    Generate images with text and background
    1. Init background images cache
    2. Load fonts
    3. Load backgrounds images list
    4. Generate sample image
        1. Generate text from wikipedia
        2. Generate background image
            1. Get random background image from cache or load new one
            2. Random crop with random color padding
            3. Convert to grayscale if needed
        3. Or generate only color background
        4. Select random font and font size
        5. Adjust font color to contrast with background
        6. Draw text on background
    """

    def __init__(
        self,
        size=(256, 256),
        min_length=5,
        max_length=30,
        backgrounds_path="backgrounds/",
        fonts_path="fonts/",
        background_ratio=0.8,
        gray_color=False,
        background_type=1,
        background_cache_size=1000,
        source="wikipedia",
        textfile="text.txt",
        debug=False,
    ):
        """
        Generate images with text and background.

        Parameters:
        - size: Tuple[int, int] - The size of the generated images.
        - min_length: int - The minimum length of the generated text.
        - max_length: int - The maximum length of the generated text.
        - backgrounds_path: str - The path to the directory containing background images.
        - fonts_path: str - The path to the directory containing font files.
        - background_ratio: float - The ratio of background images to be used.
        - gray_color: bool - Whether to convert the background images to grayscale.
        - background_type: int - The type of background to generate.
        - background_cache_size: int - The size of the background images cache.
        - source: str - The source of the text to generate.
        - textfile_path: str - The path to the text file containing the text to generate.

        Attributes:
        - backgrounds: List[str] - The list of background image file paths.
        - fonts: Dict[str, str] - The dictionary of font names and their corresponding file paths.
        - fonts_cache: Dict[str, ImageFont] - The cache of loaded font objects.
        - backgrounds_cache: Dict[str, Image] - The cache of loaded background images.
        - text_cache: List[str] - The cache of generated text strings.
        - resizer: ResizeWithPad - The image resizer object.

        Methods:
        - load_backgrounds(): Loads the background images from the specified directory.
        - load_fonts(): Loads the font files from the specified directory.
        - get_random_font(): Returns a random font object from the loaded fonts.
        - generate_image(): Generates an image with text and background.
        - get_font_color(): Calculates the font color to contrast with the background.
        - generate_text(): Generates random text from the specified source.
        - random_crop_with_padding(): Performs a random crop of the image with padding.
        - get_random_background(): Returns a random background image from the cache or loads a new one.

        Example usage:
        generator = FontGenerator(size=(256, 256), min_length=5, max_length=30, backgrounds_path='backgrounds/', fonts_path='fonts/', background_ratio=0.8, gray_color=False, background_type=1, background_cache_size=1000, source='wikipedia', textfile_path='text.txt')
        image = generator.generate_image(text='Hello World', font_size=32, font_color=(0, 0, 0), position='center', padding=10, background_image=True)
        image.show()
        """
        self.size = size
        self.min_length = min_length
        self.max_length = max_length
        self.backgrounds_path = backgrounds_path
        self.fonts_path = fonts_path
        self.background_ratio = background_ratio
        self.background_type = background_type
        self.background_cache_size = background_cache_size
        self.gray_color = gray_color
        self.source = source
        self.textfile_path = textfile

        self.backgrounds = []
        self.fonts = {}
        self.fonts_cache = {}
        self.blacklisted_fonts = []

        self.debug = debug

        # Init background images cache
        self.load_backgrounds()
        if not self.backgrounds:
            raise FileNotFoundError(
                f"No background images found under {self.backgrounds_path}"
            )

        self.load_blacklisted_fonts("blacklisted_fonts.txt")

        self.load_fonts(self.fonts_path)
        if not self.fonts:
            raise FileNotFoundError(f"No fonts found under {self.fonts_path}")

        self.resizer = ResizeWithPad(self.size, (255, 255, 255))

    def load_backgrounds(self):
        self.backgrounds = []
        for file in os.listdir(self.backgrounds_path):
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".webp"):
                self.backgrounds.append(os.path.join(self.backgrounds_path, file))

        # Create a cache for background images
        self.backgrounds_cache = {}
        self.text_cache = []

    def load_blacklisted_fonts(self, path: str):
        # load blacklisted fonts
        with open(path, "r") as f:
            for line in f:
                self.blacklisted_fonts.append(line.strip())

    def load_fonts(self, path: str):
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".ttf"):
                    if file in self.blacklisted_fonts:
                        continue
                    fontname = os.path.splitext(file)[0]
                    print(fontname, os.path.join(root, file))
                    self.fonts[fontname] = os.path.join(root, file)

    def get_random_font(self):
        font_name = random.choice(list(self.fonts.keys()))
        font_path = self.fonts[font_name]
        if font_name in self.fonts_cache:
            font = self.fonts_cache[font_name]
        else:
            font = ImageFont.truetype(font_path, size=32)
            self.fonts_cache[font_name] = font
        return font, font_name

    def generate_image(
        self,
        text,
        font_size: int = 32,
        font_color: Optional[Tuple[int, int, int]] = (0, 0, 0),
        position: str = "center",  # center, random
        padding=10,
        background_image: bool = False,
        background_color: Optional[Tuple[int, int, int]] = None,
    ) -> Image:
        logger.debug(f"Generating image with text: {text}")
        # Generate image
        if background_image:
            image = self.get_random_background()
            logger.debug(f"Background image with size: {image.size}")
            colors = get_common_colors(np.array(image), colors=12, max_points=1e5, N=1)
            logger.debug(f"Common colors: {colors}")
            main_color = colors[0]
            if font_color is None:
                candidates = [
                    opposite_color_hls(main_color),
                    *triadic_color_hls(main_color),
                ]
                font_color = random.choice(candidates)
            logger.debug(f"Font color: {font_color}")
        elif background_color is not None:
            image = Image.new("RGB", self.size, background_color)
            logger.debug(f"Background color: {background_color}")
        else:
            rand_color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            # Generate random color background
            image = Image.new("RGB", self.size, rand_color)
            logger.debug(f"Random color background: {rand_color}")

        draw = ImageDraw.Draw(image)

        # Select random font and font size
        font, font_name = self.get_random_font()
        font = font.font_variant(size=font_size)

        if font_color is None:
            # Adjust font color to contrast with background
            font_color = self.get_font_color(image)

        # Calculate position
        bbox = font.getbbox(text)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if position == "center":
            x = (self.size[0] - text_w) / 2
            y = (self.size[1] - text_h) / 2
        elif position == "random":
            # apply padding
            x = random.randint(padding, max(padding, self.size[0] - text_w - padding))
            y = random.randint(padding, max(padding, self.size[1] - text_h - padding))
        else:
            raise ValueError(f"Unknown position: {position}")

        # Draw text
        draw.text((x, y), text, fill=font_color, font=font)

        return image, font_name, font_color

    def get_font_color(self, image):
        """
        Calculate font color to contrast with background
        """
        pass

    def generate_text(self):
        """
        Generate random text from wikipedia
        """
        if len(self.text_cache) == 0:
            if self.source == "wikipedia":
                # Load text from wikipedia
                self.text_cache.extend(
                    create_strings_from_wikipedia(
                        self.min_length, 1000, "en", self.max_length
                    )
                )
            elif self.source == "textfile":
                # Load text from text file
                with open(self.textfile_path, "r") as f:
                    self.text_cache.extend(f.readlines())
                    if not self.text_cache:
                        raise ValueError(f"Text file {self.textfile_path} is empty.")

        return self.text_cache.pop()

    def random_crop_with_padding(self, image, pad_color=(255, 255, 255)):
        """
        Random crop with padding
        """
        assert image.size[0] >= self.size[0] and image.size[1] >= self.size[1]
        x = random.randint(0, image.size[0] - self.size[0])
        y = random.randint(0, image.size[1] - self.size[1])

        image = image.crop((x, y, x + self.size[0], y + self.size[1]))

        image = self.resizer(np.array(image), padding_color=pad_color)
        image = Image.fromarray(image)

        return image

    def get_random_background(self, pad_color=(255, 255, 255)):
        """
        Load background image from background cache
        """
        # Get random background image
        random_background = random.choice(self.backgrounds)

        # Load image from cache
        if random_background in self.backgrounds_cache:
            background = self.backgrounds_cache[random_background]
        else:
            background = Image.open(random_background)
            background = background.convert("RGB")
            self.backgrounds_cache[random_background] = background

        # Random crop with padding
        background = self.random_crop_with_padding(background, pad_color)

        # Apply color
        if self.gray_color:
            background = background.convert("L")

        return background


def get_n_max_logits(arr: np.array, n: int):
    """
    Get n max logits from array, return indices and values
    """
    indices = np.argpartition(arr, -n)[-n:]
    indices = indices[np.argsort(-arr[indices])]
    values = arr[indices]
    return indices, values


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("N", type=int, help="Number of generated examples")
    parser.add_argument(
        "--min_length", type=int, default=5, help="Minimum length of generated text"
    )
    parser.add_argument(
        "--max_length", type=int, default=30, help="Maximum length of generated text"
    )
    parser.add_argument("--batch_size", type=int, default=200, help="Batch size")
    parser.add_argument(
        "--max_fonts", type=int, default=3000, help="Maximum number of fonts to use"
    )
    parser.add_argument("--output", type=str, default="output/", help="Output folder")
    parser.add_argument(
        "--backgrounds",
        type=str,
        default="sample_data/backgrounds/",
        help="Path for background images, supports JPG, PNG",
    )
    parser.add_argument(
        "--fonts",
        type=str,
        default="sample_data/fonts/",
        help="Path to folder with fonts in TTF format",
    )
    parser.add_argument(
        "--font_size_min", type=int, default=16, help="Minimum font size"
    )
    parser.add_argument(
        "--font_size_max", type=int, default=96, help="Maximum font size"
    )
    parser.add_argument(
        "--background_ratio",
        type=float,
        default=0.8,
        help="Ratio between results with background image and white color",
    )
    parser.add_argument(
        "--contrast_color_ratio",
        type=float,
        default=0.5,
        help="Ratio between results with contrast color and black color",
    )
    parser.add_argument(
        "--text_source",
        type=str,
        default="wikipedia",
        help="Text source: wikipedia, textfile",
    )
    parser.add_argument(
        "--textfile",
        type=str,
        default="sample_data/textfile.txt",
        help="Path to text file with sentences dataset",
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode")

    args = parser.parse_args()
    return args


def main(args):
    # Create output folder
    os.makedirs(args.output, exist_ok=True)

    # Enable debug logger level if debug mode is on
    if args.debug:
        logger.add(sys.stdout, level="DEBUG")

    # Init font generator
    font_generator = FontGenerator(
        size=(256, 256),
        min_length=args.min_length,
        max_length=args.max_length,
        backgrounds_path=args.backgrounds,
        fonts_path=args.fonts,
        background_ratio=args.background_ratio,
        source=args.text_source,
        textfile=args.textfile,
    )

    # Generate images
    for i in tqdm(range(args.N)):
        try:
            text = font_generator.generate_text()

            if np.random.rand() < args.contrast_color_ratio:
                font_color = None
            else:
                font_color = (0, 0, 0)

            font_size = random.randint(args.font_size_min, args.font_size_max)

            if random.random() < args.background_ratio:
                background_image = True
                background_color = None
            else:
                background_image = False
                background_color = tuple(np.random.randint(0, 256, 3))

            # Generate image
            image, font_name, font_color = font_generator.generate_image(
                text,
                position="random",
                background_image=background_image,
                font_size=font_size,
                padding=10,
                font_color=font_color,
                background_color=background_color,
            )

            # Save image
            (Path(args.output) / font_name).mkdir(exist_ok=True)
            image.save(os.path.join(args.output, font_name, f"{i}.jpg"))
        except Exception as e:
            print(f"Error while generating image {i}: {e}")
            traceback.print_exc()
            continue


if __name__ == "__main__":
    args = parse_args()
    main(args)
