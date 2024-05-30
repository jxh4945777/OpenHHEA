import json
import numpy as np


class ProcessConfigs:
    def __init__(self, method_type, **kwargs):
        self.method_type = method_type
        self.__dict__.update(kwargs)

    def to_dict(self):
        return_dict = {}
        for attr, value in vars(self).items():
            if attr == "method_type":
                continue
            return_dict[attr] = value
        return return_dict

    def __str__(self):
        return f"Process Method {self.method_type} Configs :\n{json.dumps(self.to_dict(), ensure_ascii=False, indent=4)}\n"


class LoadProcessConfigs(ProcessConfigs):
    def __init__(self, 
        embedding_paths=[],
        embeddings:np.ndarray=None
    ):
        super().__init__(
            method_type="load",
            embedding_paths=embedding_paths,
            embeddings=embeddings
        )


class BertProcessConfigs(ProcessConfigs):
    def __init__(self, 
        bert_model_path:str="albert-base-v2",
        device:int=0
    ):
        super().__init__(
            method_type="albert",
            bert_model_path=bert_model_path,
            device=device
        )


class CLIPProcessConfigs(ProcessConfigs):
    def __init__(self, 
        clip_model_path:str=""
    ):
        super().__init__(
            method_type="clip",
            clip_model_path=clip_model_path
        )


class FualignProcessConfigs(ProcessConfigs):
    def __init__(self, 
        weighted:bool=False,
        directed:bool=False,
        walk_length:int=80,
        num_walks:int=10,
        p:float=1e-100,
        q:float=1.0,
        dim:int=64,
        window_size:int=10,
        workers:int=12,
        iter:int=5
    ):
        super().__init__(
            method_type="fualign",
            weighted = weighted,
            directed = directed,
            walk_length = walk_length,
            num_walks = num_walks,
            p = p,
            q = q,
            dim = dim,
            window_size = window_size,
            workers = workers,
            iter = iter
        )