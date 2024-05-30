import gc
import math
import numpy as np
from tqdm import tqdm

import torch
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, CLIPModel

from OpenHHEA.configs.types import ProcessName
from OpenHHEA.data import KGDataLoader
from .utils_dataprocess import normalize_embed_dict



class NameProcessor:
    def __init__(self) -> None:
        pass

    def name_process(self, loader:KGDataLoader) -> np.ndarray:
        pass


class BertNameProcessor(NameProcessor):
    def __init__(self, 
        dim:int=64, 
        bert_model_path:str="albert-base-v2", 
        device:int=0
    ) -> None:
        super().__init__()
        self.dim = dim
        self.device = f"cuda:{device}" if device >=0 and torch.cuda.is_available() else "cpu"
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)
        self.model = BertModel.from_pretrained(bert_model_path).to(self.device)

    def get_bert_embedding(self, text:str) -> np.ndarray:
        # preprocess
        text = text.replace("\t", "").replace("_", " ").replace("/", " ")
        # get embedding
        inputs = self.tokenizer(text, return_tensors="pt")
        for key in inputs.keys():
            inputs[key] = inputs[key].to(self.device)
        outputs = torch.mean(self.model(**inputs).last_hidden_state, dim=1)
        embedding = outputs[0].detach().cpu().numpy()
        return np.array(embedding, dtype=np.float32)

    def name_process(self, loader:KGDataLoader) -> np.ndarray:
        ent_num = loader.get_num_of_entity()
        name_embed = {}
        for eid, name in loader.ent_id2origin_name.items():
            name_embed[eid] = self.get_bert_embedding(name)
        embeddings = normalize_embed_dict(ent_num, name_embed)
        ### calculate kernel and bias
        u, s, _ = np.linalg.svd(np.cov(embeddings.T))
        W = np.dot(u, np.diag(1 / np.sqrt(s)))
        kernel = W[:, :self.dim]
        bias:np.ndarray = embeddings.mean(axis=0, keepdims=True)
        ### transform and normalize embeddings
        embeddings = (embeddings + bias).dot(kernel)
        embeddings = embeddings / (embeddings ** 2).sum(axis=1, keepdims=True)**0.5
        gc.collect()
        return embeddings


class CLIPNameProcessor(NameProcessor):
    def __init__(self, 
        clip_model_path:str, 
        batch_size:int=2048
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(clip_model_path)
        self.model = CLIPModel.from_pretrained(clip_model_path)

    def name_process(self, loader:KGDataLoader) -> np.ndarray:
        ### get text features
        ent_num = loader.get_num_of_entity()
        eids = np.array(list(loader.ent_id2origin_name.keys()))
        idxs = list(range(len(eids)))

        bs = self.batch_size
        batch_idxs = [idxs[i*bs : (i+1)*bs] for i in range(math.ceil(len(idxs)/bs))]
        name_embed = {}
        for batch in tqdm(batch_idxs, desc="CLIP process name"):
            batch_eids = eids[batch]
            batch_names = [loader.ent_id2origin_name[eid] for eid in batch_eids]
            inputs = self.tokenizer(batch_names, padding=True, return_tensors="pt")
            features = self.model.get_text_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            features = features.cpu().detach().numpy()
            for eid, feat in zip(batch_eids, features):
                name_embed[eid] = feat
        ### normalize embeddings
        embeddings = normalize_embed_dict(ent_num, name_embed)
        gc.collect()
        return embeddings


# unified inference function
processor_dict = {
    ProcessName.BERT : BertNameProcessor,
    ProcessName.CLIP : CLIPNameProcessor
}


def get_name_processor(method_type, **kwargs) -> NameProcessor:
    if method_type not in processor_dict:
        raise Exception(f"Error occured when process name : there is no name processor using {method_type}")
    return processor_dict[method_type](**kwargs)

def name_embedding_process(method_type, loader:KGDataLoader, **kwargs) -> np.ndarray:
    if method_type not in ProcessName:
        raise Exception(f"Error occured when process name : there is no name process method {method_type}")
    processor = get_name_processor(method_type, **kwargs)
    embeddings = processor.name_process(loader)
    gc.collect()
    return embeddings