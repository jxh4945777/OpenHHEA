import gc
import random
import numpy as np

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm

from OpenHHEA.configs.types import ProcessImage
from OpenHHEA.data import KGDataLoader
from .utils_dataprocess import normalize_embed_dict



class ImageProcessor:
    def __init__(self) -> None:
        pass

    def image_process(self, loader:KGDataLoader) -> np.ndarray:
        pass


class CLIPImageProcessor(ImageProcessor):
    def __init__(self, 
        clip_model_path:str
    ) -> None:
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained(clip_model_path)
        self.model = CLIPModel.from_pretrained(clip_model_path)
    
    def image_process(self, loader:KGDataLoader) -> np.ndarray:
        ### get image features

        image_embed = {}
        for eid, paths in tqdm(list(loader.image_paths.items()), desc="CLIP process image"):
            img_path = random.choice(paths)
            img = Image.open(img_path)
            inputs = self.processor(images=img, return_tensors="pt")
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            image_embed[eid] = features.cpu().detach().numpy().reshape(-1)
            gc.collect()
        ### normalize embeddings
        embeddings = normalize_embed_dict(loader.get_num_of_entity(), image_embed)
        return embeddings



### unified inference function
processor_dict = {
    ProcessImage.CLIP : CLIPImageProcessor
}

def get_image_processor(method_type, **kwargs) -> ImageProcessor:
    return processor_dict[method_type](**kwargs)

def image_embedding_process(method_type, loader:KGDataLoader, **kwargs) -> np.ndarray:
    if method_type not in ProcessImage:
        raise Exception(f"Error occured when process image : there is no image process method {method_type}")
    processor = get_image_processor(method_type, **kwargs)
    embeddings = processor.image_process(loader)
    gc.collect()
    return embeddings
