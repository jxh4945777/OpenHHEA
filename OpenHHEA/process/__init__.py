from .dataprocessor import KGDataProcessor

from .name_process import get_name_processor, name_embedding_process
from .image_process import get_image_processor, image_embedding_process
from .structure_process import get_struct_processor, struct_embedding_process
from .utils_dataprocess import load_embeddings_from_paths