from .default_converter import DefaultConverter
from .image_converter import ImageConverter


def get_converter(module_id: str) -> ImageConverter:
    if module_id == "icasbr":
        return DefaultConverter()
    elif module_id == "icaslv":
        return DefaultConverter()
    elif module_id == "icasln":
        return DefaultConverter()
    else:
        raise ValueError(f"Unknown module_id: {module_id}")
