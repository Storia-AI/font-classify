"""
Generate dataset for font classification task
"""
import os
import random
import traceback
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops
from typing import Union, Tuple, List, Dict, Optional


import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.cluster import KMeans # the KNN Clustering module


def get_common_colors(img, colors=32, max_points=-1, N=3,
              colorspace='rgb', points_scale=(1,1), select_color='mean',
              **args):
    max_points = int(max_points)
    img = np.array(img, dtype=np.uint8)
    h, w = img.shape[0], img.shape[1]

    img_orig_flat = img.reshape(h * w, 3)

    if colorspace == 'bgr':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif colorspace == 'hls':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif colorspace == 'hsv':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif colorspace == 'lab':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    elif colorspace == 'rgb':
        img = img.copy()
    else:
        raise Exception('Unknown colorspace')

    img_flat = img.copy().reshape(h * w, 3)


    if max_points > 0 and max_points < img_flat.shape[0]:
        idx = np.random.choice(np.arange(img_flat.shape[0]), max_points, replace=False)
        kmeans = KMeans(n_clusters=colors, random_state=0).fit(img_flat[idx])
        labels = kmeans.predict(img_flat)
    else:
        kmeans = KMeans(n_clusters=colors, random_state=0).fit(img_flat)
        labels = kmeans.labels_

    unique_labels, counts = np.unique(labels, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]
    most_common_labels = unique_labels[sorted_indices[:N]]

    img_flat_out = img_orig_flat.copy()

    # loops for cluster center
    colors = []
    for ci in np.unique(most_common_labels):
        if select_color == 'mean':
            colors.append(img_orig_flat[labels==ci,:].mean(axis=0))
        elif select_color == 'median':
            colors.append(np.median(img_orig_flat[labels==ci,:], axis=0))
        elif select_color == 'mode':
            colors.append(st.mode(img_orig_flat[labels==ci,:], axis=0)[0])

    return [c.astype(np.uint8) for c in colors]

def load_image(image_path):
    return Image.open(image_path).convert('RGB')

import colorsys

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
    h_triadic1 = (h + 1/3) % 1
    h_triadic2 = (h + 2/3) % 1
    return hls_to_rgb((h_triadic1, l, s)), hls_to_rgb((h_triadic2, l, s))


def opposite_color_hls(rgb):
    h, l, s = rgb_to_hls(rgb)
    l = 1.0 - l
    h_opposite = (h + 1/2) % 1
    return hls_to_rgb((h_opposite, l, s))



class ResizeWithPad:
    
    def __init__(self, new_shape: Tuple[int, int], 
                 padding_color: Tuple[int] = (255, 255, 255)) -> None:
        self.new_shape = new_shape
    
    def __call__(self, image: np.array, padding_color, **kwargs) -> np.array:
        """Maintains aspect ratio and resizes with padding.
        Params:
            image: Image to be resized.
            new_shape: Expected (width, height) of new image.
            padding_color: Tuple in BGR of padding color
        Returns:
            image: Resized image with padding
        """
        original_shape = (image.shape[1], image.shape[0])
        ratio = float(max(self.new_shape))/max(original_shape)
        new_size = tuple([int(x*ratio) for x in original_shape])
        image = cv2.resize(image, new_size)
        delta_w = self.new_shape[0] - new_size[0]
        delta_h = self.new_shape[1] - new_size[1]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
        return image


class CutMax:
    """Cuts the image to the maximum size """
    
    def __init__(self, max_size: int = 1024) -> None:
        self.max_size = max_size
    
    def __call__(self, image: np.array, **kwargs) -> np.array:
        """Cuts the image to the maximum size """
        if image.shape[0] > self.max_size:
            image = image[:self.max_size, :, :]
        if image.shape[1] > self.max_size:
            image = image[:, :self.max_size, :]
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
    
    def __init__(self, size=(256, 256), min_length=5, max_length=30, backgrounds_path='backgrounds/', fonts_path='fonts/', background_ratio=0.8,
                 gray_color=False, background_type=1, background_cache_size=1000):
        self.size = size
        self.min_length = min_length
        self.max_length = max_length
        self.backgrounds_path = backgrounds_path
        self.fonts_path = fonts_path
        self.background_ratio = background_ratio
        self.background_type = background_type
        self.background_cache_size = background_cache_size
        self.gray_color = gray_color
        
        self.backgrounds = []
        self.fonts = []
        
        # Init background images cache
        self.load_backgrounds()
        self.load_fonts()
        
        self.resizer = ResizeWithPad(self.size, (255, 255, 255))
        
        
    def load_backgrounds(self):
        self.backgrounds = []
        for file in os.listdir(self.backgrounds_path):
            if file.endswith('.jpg') or file.endswith('.png'):
                self.backgrounds.append(os.path.join(self.backgrounds_path, file))
                
        # Create a cache for background images
        self.backgrounds_cache = {}
    
    def load_fonts(self):
        # load blacklisted fonts
        self.blacklisted_fonts = []
        with open('blacklisted_fonts.txt', 'r') as f:
            for line in f:
                self.blacklisted_fonts.append(line.strip())
        
        self.fonts = []
        for file in os.listdir(self.fonts_path):
            if file.endswith('.ttf'):
                if file in self.blacklisted_fonts:
                    continue
                self.fonts.append(os.path.join(self.fonts_path, file))
    
    def generate_image(self, text, font,
                       font_size: int = 32,
                       font_color: Optional[Tuple[int, int, int]] = (0, 0, 0),
                       position: str = 'center',  # center, random
                       padding=10,
                       background_image: bool = False,
                       background_color: Optional[Tuple[int, int, int]] = None,
                       ) -> Image:
        # Generate image
        if background_image:
            image = self.get_random_background()
            colors = get_common_colors(np.array(image), colors=24, max_points=1e5, N=1)
            main_color = colors[0]
            if font_color is None:
                candidates = [opposite_color_hls(main_color), *triadic_color_hls(main_color)]
                font_color = random.choice(candidates)
        elif background_color:
            image = Image.new('RGB', self.size, background_color)
        else:
            # Generate random color background
            image = Image.new('RGB', self.size, 
                              (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font, size=font_size)
        
        if font_color is None:
            # Adjust font color to contrast with background
            font_color = self.get_font_color(image)
        
        # Calculate position
        text_w, text_h = font.getsize(text)
        if position == 'center':
            x = (self.size[0] - text_w) / 2
            y = (self.size[1] - text_h) / 2
        elif position == 'random':
            # apply padding
            x = random.randint(padding, self.size[0] - text_w - padding)
            y = random.randint(padding, self.size[1] - text_h - padding)
        else:
            raise ValueError(f'Unknown position: {position}')
        
        # Draw text       
        draw.text((x, y), text, fill=font_color, font=font)
        
        # Apply blur
        if self.random_blur:
            blur = random.randint(0, self.blur)
        else:
            blur = self.blur
        image = image.filter(ImageFilter.GaussianBlur(blur))
        
        return image
    
    def get_font_color(self, image):
        """
        Calculate font color to contrast with background
        """
        pass
    
    def random_crop_with_padding(self, image, pad_color=(255, 255, 255)):
        """
        Random crop with padding
0        """
        assert image.size[0] >= self.size[0] and image.size[1] >= self.size[1]
        x = random.randint(0, image.size[0])
        y = random.randint(0, image.size[1])
        
        image = self.resizer(np.array(image), padding_color=pad_color)
        image = Image.fromarray(image)
        
        return image
    
    def get_random_background(self, w=256, h=256, pad_color=(255, 255, 255)):
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
            background = background.convert('RGB')
            self.backgrounds_cache[random_background] = background
        
        # Random crop with padding
        background = self.random_crop_with_padding(background, w, h, pad_color)    
        
        # Apply color
        if self.gray_color:
            background = background.convert('L')
            
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
    parser.add_argument('N', type=int, help='Number of generated examples')
    parser.add_argument('--min_length', type=int, default=5, help='Minimum length of generated text')
    parser.add_argument('--max_length', type=int, default=30, help='Maximum length of generated text')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch size')
    parser.add_argument('--max_fonts', type=int, default=3000, help='Maximum number of fonts to use')
    parser.add_argument('--output', type=str, default='output/', help='Output folder')
    parser.add_argument('--backgrounds', type=str, default='backgrounds/', help='Path for background images, supports JPG, PNG')
    parser.add_argument('--fonts', type=str, default='fonts/', help='Path to folder with fonts in TTF format')
    parser.add_argument('--background_ratio', type=float, default=0.8, help='Ratio between results with background image and white color')

    args = parser.parse_args()
    return args


def main(args):
    # Init font generator
    font_generator = FontGenerator(size=(256, 256), min_length=args.min_length, max_length=args.max_length, backgrounds_path=args.backgrounds, fonts_path=args.fonts, background_ratio=args.background_ratio)
    
    # Generate images
    for i in range(args.N):
        try:
            text = font_generator.generate_text()
            # Generate image
            image = font_generator.generate_image(text, font, background)
            
            # Save image
            image.save(os.path.join(args.output, f'{i}.jpg'))
        except Exception as e:
            print(f'Error while generating image {i}: {e}')
            traceback.print_exc()
            continue


if __name__ == "__main__":
    args = parse_args()
    main(args)