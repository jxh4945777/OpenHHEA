import os
import torch
import random
import numpy as np
from OpenHHEA.configs.types import *



### basic config
class ModelConfig:
    def __init__(self, method_type, hit_k:list=[1, 5, 10], device:int=0, random_seed:int=None):
        self.has_set_config = False
        self.method_type = method_type
        self.hit_k = hit_k
        self.device = f"cuda:{device}" if torch.cuda.is_available() and device > 0 else "cpu"
        self.random_seed = random_seed
        if self.random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed(random_seed)
    
    def set_config(self, **kwargs):
        self.has_set_config = True
        self.__dict__.update(kwargs)

    def to_dict(self):
        return_json = {}
        for attr, value in vars(self).items():
            if isinstance(value, list):
                if not isinstance(value, dict):
                    value = "[" + ", ".join([str(v) for v in value]) + "]"
            if attr not in ["has_set_config"]:
                return_json[attr] = value
            if attr == "method_type":
                return_json[attr] = value.value
        return return_json

    def __str__(self):
        output = "Configs :\n"
        for attr, value in vars(self).items():
            if isinstance(value, dict):
                output += f"\t{attr} : {{\n"
                for k, v in value.items():
                    output += f"\t    {k} : {v}\n"
                output += "\t}\n"
            else:
                output += f"\t{attr} : {value}\n"
        return output


class EmbeddingBasedConfig(ModelConfig):
    def __init__(self, 
        method_type, 
        hit_k:list=[1, 5, 10], 
        device:int=0, 
        random_seed:int=None,
        ### train configs
        gamma:float=1.0,
        learning_rate:float=0.01,
        weight_decay:float=0.001,
        epochs:int=1500,
        batch_size:int=1024,
        iter_turn:int=5,
        neg_sample:bool=False,
        ### eval configs
        csls:int=10,
        eval_step:int=10,
        eval_threads:int=16,
        model_save_dir:str=None
    ):
        super().__init__(method_type, hit_k, device, random_seed)
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.iter_turn = iter_turn
        self.neg_sample = neg_sample

        self.csls = csls
        self.eval_step = eval_step
        self.eval_threads = eval_threads
        self.model_save_dir = model_save_dir


class LLMBasedConfig(ModelConfig):
    def __init__(self, 
        method_type, 
        hit_k:list=[1, 5, 10], 
        device:int=0, 
        random_seed:int=None,
        ### LLM base configs
        api_key:str=None,
        api_base:str=None,
        api_model:str=None,
        ### LLM request configs
        history_len:int=3,
        print_log:bool=True
    ):
        super().__init__(method_type, hit_k, device, random_seed)
        self.api_key = api_key
        self.api_base = api_base
        self.api_model = api_model
        self.history_len = history_len
        self.print_log = print_log

    def set_config(self, 
        ### LLM base configs
        api_key:str,
        api_base:str,
        api_model:str,
        ### LLM request configs
        history_len:int=3,
        print_log:bool=True,
    ):
        super().set_config()
        self.api_key = api_key
        self.api_base = api_base
        self.api_model = api_model
        
        self.history_len = history_len
        self.print_log = print_log


### config for SimpleHHEA
class SimpleHHEAConfig(EmbeddingBasedConfig):
    def __init__(self,
        device:int=0,
        hit_k:list=[1, 5, 10],
        random_seed:int=None,
        ### hyperparameter
        name_noise_ratio:float=0.0,
        emb_size:int=64,
        structure_size:int=8,
        time_size:int=8,
        ### ablation configs
        use_structure:bool=True,
        use_time:bool=True,
        ### train configs
        gamma:float=1.0,
        lr:float=0.01,
        weight_decay:float=0.001,
        epochs:int=1500,
        csls:int=10,
        eval_step:int=10,
        eval_threads:int=16,
        model_save_dir:str=None
    ):
        super().__init__(
            method_type=Methods.SIMPLE_HHEA, 
            hit_k=hit_k, 
            device=device, 
            random_seed=random_seed,
            gamma=gamma,
            learning_rate=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=0,
            iter_turn=1,
            neg_sample=True,
            csls=csls,
            eval_step=eval_step,
            eval_threads=eval_threads,
            model_save_dir=model_save_dir
        )
        ### model configs 
        self.name_noise_ratio = name_noise_ratio
        self.emb_size = emb_size
        self.structure_size = structure_size
        self.time_size = time_size
        self.use_structure = use_structure
        self.use_time = use_time

    def set_config(self,
        device:int=0,
        ### hyperparameter
        name_noise_ratio:float=0.0,
        emb_size:int=64,
        structure_size:int=8,
        time_size:int=8,
        ### ablation configs
        use_structure:bool=True,
        use_time:bool=True,
        ### train configs
        gamma:float=1.0,
        lr:float=0.01,
        weight_decay:float=0.001,
        epochs:int=1500,
        csls:int=10,
        eval_step:int=10,
        eval_threads:int=16,
        model_save_dir:str=None
    ):
        '''
        SimpleHHEA Parameters
        ------------
        0. basic configs
        device  : int
            the device to run the model
        
        1. hyperparameter
        name_noise_ratio    : float
            the ratio of noise in name embedding
        emb_size        : int
            the size of name encoder layer
        structure_size  : int
            the size of structure encoder layer
        time_size       : int
            the size of time encoder layer
        
        2. ablation configs
        use_structure   : bool
            whether to use structure embedding
        use_time    : bool
            whether to use time embedding
        
        3. train configs
        gamma   : float
            gamma in L1 loss
        lr      : float
            learning rate
        weight_decay    : float
            weight decay of optimizer
        epochs  : int
            train epochs
        csls  : int
            CSLS paramters
        eval_step   : int
            how many steps to check the results of training
        eval_threads    : int
            number of threads when evaluating
        model_save_dir : str or None
            the path where to save the trained model
        '''
        super().set_config()
        self.device = f"cuda:{device}" if torch.cuda.is_available() and device > 0 else "cpu"
        ### hyperparamter 
        self.name_noise_ratio = name_noise_ratio
        self.emb_size = emb_size
        self.structure_size = structure_size
        self.time_size = time_size
        ### ablation configs
        self.use_structure = use_structure
        self.use_time = use_time
        ### train configs
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.epochs = epochs
        self.csls = csls
        self.eval_step = eval_step
        self.eval_threads = eval_threads
        self.model_save_dir = model_save_dir
        if self.model_save_dir is not None:
            if not os.path.exists(self.model_save_dir):
                os.makedirs(self.model_save_dir)


### config for XGEA
class XGEAConfig(EmbeddingBasedConfig):
    def __init__(self,
        device:int=0,
        hit_k:list=[1, 5, 10],
        random_seed:int=None,
        ### hyperparameter
        embed_noise_ratio:float=0.0,
        node_hidden_size:int=128,
        rel_hidden_size:int=128,
        depth:int=2,
        dropout_rate:float=0.3,
        ### ablation configs
        use_literal:bool=False,
        ### train configs
        unsup_k:int=6300,
        batch_size:int=1024,
        gamma:float=1.0,
        lr:float=0.01,
        weight_decay:float=0.0005,
        epochs:int=50,
        turn:int=5,
        csls:int=10,
        eval_step:int=10,
        eval_threads:int=16,
        model_save_dir:str=None
    ):
        super().__init__(
            method_type=Methods.XGEA, 
            hit_k=hit_k, 
            device=device, 
            random_seed=random_seed, 
            gamma=gamma,
            learning_rate=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            iter_turn=turn,
            neg_sample=False,
            csls=csls,
            eval_step=eval_step,
            eval_threads=eval_threads,
            model_save_dir=model_save_dir    
        )
        ### model configs 
        self.embed_noise_ratio = embed_noise_ratio
        self.node_hidden_size = node_hidden_size
        self.rel_hidden_size = rel_hidden_size
        self.dropout_rate = dropout_rate
        self.depth = depth
        ### ablation configs
        self.use_literal = use_literal
        ### train configs
        self.unsup_k = unsup_k

    def set_config(self,
        device:int=0,
        ### hyperparameter
        embed_noise_ratio:float=0.0,
        node_hidden_size:int=128,
        rel_hidden_size:int=128,
        depth:int=2,
        dropout_rate:float=0.3,
        ### ablation configs
        use_literal:bool=False,
        ### train configs
        unsup_k:int=6300,
        batch_size:int=1024,
        gamma:float=1.0,
        lr:float=0.01,
        weight_decay:float=0.0005,
        epochs:int=50,
        turn:int=5,
        csls:int=10,
        eval_step:int=10,
        eval_threads:int=16,
        model_save_dir:str=None
    ):
        '''
        XGEA Parameters
        ------------
        0. basic configs
        device  : int
            the device to run the model
        
        1. hyperparameter
        embed_noise_ratio    : float
            the ratio of noise in name embedding
        node_hidden_size    : int
            size of hidden layer for node
        rel_hidden_size     : int
            size of hidden layer for relation
        
        2. ablation configs
        use_literal   : bool
            whether to use literal information
        
        3. train configs
        batch_size  : int
            batch size when training
        dropout_rate    : float
            rate of dropout
        gamma   : float
            gamma in L1 loss
        lr      : float
            learning rate
        weight_decay    : float
            weight decay of optimizer
        epochs  : int
            number of training epochs
        turn    : int
            number of iteration turns
        depth   : int
            depth of GCN layers
        unsup_k : int

        csls  : int
            CSLS paramters
        eval_step   : int
            how many steps to check the results of training
        eval_threads    : int
            number of threads when evaluating
        model_save_dir : str or None
            the path where to save the trained model
        '''
        super().set_config()
        self.device = f"cuda:{device}" if torch.cuda.is_available() and device > 0 else "cpu"
        ### hyperparamter 
        self.embed_noise_ratio = embed_noise_ratio
        self.node_hidden_size = node_hidden_size
        self.rel_hidden_size = rel_hidden_size
        self.dropout_rate = dropout_rate
        self.depth = depth
        ### ablation configs
        self.use_literal = use_literal
        ### train configs
        self.batch_size = batch_size
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.epochs = epochs
        self.turn = turn
        self.unsup_k = unsup_k
        self.csls = csls
        self.eval_step = eval_step
        self.eval_threads = eval_threads
        self.model_save_dir = model_save_dir
        if self.model_save_dir is not None:
            if not os.path.exists(self.model_save_dir):
                os.makedirs(self.model_save_dir)



### config for llm_chatalign
DEFAULT_WEIGHTS = {
    SimDimension.NAME : 0.5,
    SimDimension.DESC : 0.3,
    SimDimension.STRUCT : 0.2,
    SimDimension.TIME : 0.1,
    SimDimension.IMAGE : 0.1
}

class LLMChatAlignConfig(LLMBasedConfig):
    def __init__(self,
        device:int=0,
        hit_k:list=[1, 5, 10],
        random_seed:int=None,
        ### LLM base configs
        api_key:str=None,
        api_base:str=None,
        api_model:str=None,
        ### LLM request configs
        history_len:int=3,
        print_log:bool=True,
        ##### LLMChatAlign configs
        mmea_sim_alpha:float=0.3,
        noise_ratio:float=0.0,
        data_sample_num:int=200,
        candidates_num:int=20,
        neighbor_num:int=10,
        threshold:float=0.5,
        weights:dict=DEFAULT_WEIGHTS,
        search_range:list=[1, 10, 20],
        ##### ablation configs
        use_code:bool=True,
        use_desc:bool=True,
        use_name:bool=True,
        use_struct:bool=True,
        use_img:bool=True,
        use_time:bool=True,
    ):
        super().__init__(Methods.LLMCHATALIGN, hit_k, device, random_seed, api_key, api_base, api_model, history_len)

        self.alpha = mmea_sim_alpha
        self.noise_ratio = noise_ratio
        self.data_sample_num = data_sample_num
        self.candidates_num = candidates_num
        self.neighbor_num = neighbor_num
        self.threshold = threshold
        self.weights = weights
        self.search_range = search_range

        self.use_code = use_code
        self.use_desc = use_desc
        self.use_name = use_name
        self.use_struct = use_struct
        self.use_img = use_img
        self.use_time = use_time

        self.print_log = print_log

    ### API
    def set_config(self,
        ##### LLM base configs
        api_key:str, 
        api_model:str,
        api_base:str,
        ##### LLM request configs
        history_len:int=3,
        print_log:bool=True,
        ##### LLMChatAlign configs
        mmea_sim_alpha:float=0.3,
        noise_ratio:float=0.0,
        data_sample_num:int=200,
        candidates_num:int=20,
        neighbor_num:int=10,
        threshold:float=0.5,
        weights:dict=DEFAULT_WEIGHTS,
        search_range:list=[1, 10, 20],
        ##### ablation configs
        use_code:bool=True,
        use_desc:bool=True,
        use_name:bool=True,
        use_struct:bool=True,
        use_img:bool=True,
        use_time:bool=True,
    ):
        '''
        LLM-ChatAlign Parameters
        ------------
        1. base configs of LLM
        api_key     : str
            the api key of LLM
        api_model   : str
            the model of LLM
        api_base    : str
            base_url of LLM. If None, use default openai base_url
        
        2. request configs of LLM
        history_len     : int
            the max length of LLM's chat history
        print_log   : bool
            whether to print the outputs of LLM

        3. configs of LLMChatAlign pipeline
        mmea_sim_alpha : float
            the weight of mmea_sim when get candidates
        noise_ratio    : float
            the ratio of noise in name embedding
        data_sample_num : int
            the amount of data sampled from the test set
        candidates_num  : int
            the amount of candidate entities for each main entity
        neighbor_num    : int
            the amount of neighbor entities for each entity
        threshold       : int
            the threshold of using LLM to rerank candidates
        weights         : dict
            the weight of each similarity
        
        4. configs of ablation study
        use_code    : bool
            whether to use code format to represent Knowledge Graph
        use_desc    : bool
            whether to use description information
        use_name    : bool
            whether to use name information
        use_struct  : bool
            whether to use structure information
        use_img     : bool
            whether to use image information
        use_time    : bool
            whether to use time information
        '''
        super().set_config(api_key, api_base, api_model, history_len, print_log)
        self.alpha = mmea_sim_alpha
        self.noise_ratio = noise_ratio
        self.data_sample_num = data_sample_num
        self.candidates_num = candidates_num
        self.neighbor_num = neighbor_num
        self.threshold = threshold
        self.weights = weights
        self.search_range = search_range

        self.use_code = use_code
        self.use_desc = use_desc
        self.use_name = use_name
        self.use_struct = use_struct
        self.use_img = use_img
        self.use_time = use_time


def get_model_config(method_type):
    if isinstance(method_type, Enum):
        if method_type not in Methods:
            raise Exception(f"Error occured when get model config : there is no method named {method_type}")
        if method_type == Methods.SIMPLE_HHEA:
            return SimpleHHEAConfig()
        if method_type == Methods.LLMCHATALIGN:
            return LLMChatAlignConfig()
    elif isinstance(method_type, str):
        if method_type not in Methods._value2member_map_:
            raise Exception(f"Error occured when get model config : there is no method named {method_type}")
        if method_type == Methods.SIMPLE_HHEA.value:
            return SimpleHHEAConfig()
        if method_type == Methods.LLMCHATALIGN.value:
            return LLMChatAlignConfig()
