from typing import List
from .image_converter import ImageConverter


class DefaultConverter(ImageConverter):
    def _select_best_image(self, output_images_paths: List[str]) -> str:
        # for now, just return the first image in the list
        return output_images_paths[0]
