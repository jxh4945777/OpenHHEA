# OpenHHEA :  Toolkit for Entity Alignment

Code of OpenHHEA



### Requirements

```
python >= 3.9
scipy
numpy
networkx
pillow
gensim
openai
transformers
torch
```



### Main Interface

The main interface provided by OpenHHEA is included as follows:

- `OpenHHEA.get_model_config`: get the predifined configurations of model and data processing. 
- `OpenHHEA.KGDataLoader`: the data loader for loading knowledge graph data.
- `OpenHHEA.KGDataProcessor`: the processor for process the knowledge graph data to get feature embeddings of name information, structure information, image information and so on.
- `OpenHHEA.HHEATrainer`: the trainer for training embedding-based entity alignment model. Use `train` function to start training.
- `OpenHHEA.ReasoningXXX`: the encapsulated entity alignment reasoning process. Use `run` function to start entity alignment reasoning.
- `OpenHHEA.init_model_pipeline`: get the predefined entity alignment pipeline, including the training and reasoning. Use `run` to start the whole entity alignment pipeline.



### Usage

For the detailed usage of OpenHHEA, including the methods for using customized reasoning workflow and using redefined reasoning workflow, please see [example_code.py](./example_code.py)



