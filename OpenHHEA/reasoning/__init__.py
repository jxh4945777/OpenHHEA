from .models import *
from OpenHHEA.configs.model_configs import *
from OpenHHEA.configs.types import *
from OpenHHEA.data import *
from OpenHHEA.process import *

from .reason_tools import ReasoningEmbeddingBased
        

def init_model_pipeline(method_type:Enum, config:ModelConfig, dataloader:KGDataLoader, dataprocessor:KGDataProcessor):
    if method_type not in Methods:
        raise Exception(f"Error occured when get model pipeline : there is no method named {method_type}")
    if method_type == Methods.SIMPLE_HHEA:
        return PipelineSimpleHHEA(
            config=config,
            dataloader=dataloader,
            dataprocessor=dataprocessor
        )
    if method_type == Methods.LLMCHATALIGN:
        return PipelineLLMChatAlign(
            config=config,
            dataloader=dataloader,
            dataprocessor=dataprocessor
        )
    if method_type == Methods.XGEA:
        return PipelineXGEA(
            config=config,
            dataloader=dataloader,
            dataprocessor=dataprocessor
        )