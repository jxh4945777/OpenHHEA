import os
import time
import openai
import pickle
import numpy as np

import torch
import torch.nn as nn

from OpenHHEA.configs.model_configs import EmbeddingBasedConfig, LLMBasedConfig
from OpenHHEA.data import KGDataLoader
from OpenHHEA.process import KGDataProcessor
from OpenHHEA.evaluate import eval_alignment_by_embedding, eval_alignment_by_ranks
from .results import ReasoningResult


### Pipeline of Embedding-Based Method
class ReasoningEmbeddingBased:
    def __init__(self,
        config:EmbeddingBasedConfig,
        ### data 
        dataloader:KGDataLoader, 
        dataprocessor:KGDataProcessor, 
        ### model
        model:nn.Module, 
        model_path:str=None,
    ) -> None:
        self.config = config
        self.loader = dataloader
        self.processor = dataprocessor
        self.entity_embeddings = None

        self.model = model.to(self.config.device)

        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path, map_location=self.config.device))

        self.best_model_path = os.path.join(self.config.model_save_dir, f"{self.config.method_type.value}_best_model.pth")

    def run(self, save_result_path:str=None, **kwargs):
        if not self.config.has_set_config:
            raise Exception(f"Model config hasn't been set !")
        st = time.time()
        ### update embeddings
        self.update_entity_embeddings(**kwargs)
        ### evaluate
        feat = self.entity_embeddings(self.loader.ref_pairs)
        lvec, rvec = feat[:, 0, :], feat[:, 1, :]
        lvec = lvec / np.linalg.norm(lvec, axis=-1, keepdims=True)
        rvec = rvec / np.linalg.norm(rvec, axis=-1, keepdims=True)
        t_prec_set, acc, t_mrr = eval_alignment_by_embedding(lvec, rvec, self.config.hit_k, self.config.eval_threads, self.config.csls, output=True)
        result = ReasoningResult(
            config=self.config,
            hits=acc,
            mrr=t_mrr,
            time_cost=time.time() - st
        )
        if save_result_path is not None:
            save_dir = os.path.split(save_result_path)[0]
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open(save_result_path, "a+", encoding="utf-8") as fw:
                fw.write("\n" + str(result) + "\n")
        return result
    
    def update_entity_embeddings(self, **kwargs):
        self.model.eval()
        with torch.no_grad():
            embeddings:torch.Tensor = self.model(**kwargs)
            self.entity_embeddings = embeddings.detach().cpu().numpy()
        return self.entity_embeddings

    def save_entity_embeddings(self, save_path:str):
        save_dir = os.path.split(save_path)[0]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if save_path.endswith(".txt"):
            np.savetxt(save_path, self.entity_embeddings)
        elif save_path.endswith(".npy"):
            np.save(save_path, self.entity_embeddings)
        else:
            embed_dict = {i:embed for i, embed in enumerate(self.entity_embeddings)}
            with open(save_path, "wb") as fw:
                pickle.dump(embed_dict, fw)
