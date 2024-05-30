from enum import Enum


### Types for data processing



### Types for reasoning process
class Methods(Enum):
    SIMPLE_HHEA = "SimpleHHEA"
    XGEA = "XGEA"
    
    LLMCHATALIGN = "LLMChatAlign"


### Types for data process
class ProcessName(Enum):
    LOAD = "load"
    BERT = "bert"
    CLIP = "clip"

class ProcessImage(Enum):
    LOAD = "load"
    CLIP = "clip"

class ProcessStruct(Enum):
    LOAD = "load"
    FUALIGN = "fualign"

class ProcessEntity(Enum):
    LOAD = "load"


### Types for specific model
###### Types for LLM_ChatAlign
class SimDimension(Enum):
    NAME = "name"
    DESC = "description"
    STRUCT = "structure"
    IMAGE = "image"
    TIME = "time"