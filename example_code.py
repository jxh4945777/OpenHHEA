# encoding = utf-8
import os
import OpenHHEA
from OpenHHEA.configs.process_configs import *
from OpenHHEA.reasoning.results import SimpleHHEAResult


def customized_reasoning(
    configs:OpenHHEA.configs.model_configs.SimpleHHEAConfig, 
    loader:OpenHHEA.KGDataLoader, 
    processor:OpenHHEA.KGDataProcessor,
    result_path:str
) -> SimpleHHEAResult:
    '''
    Example code for entity alignment using a flexible customized reasoning workflow.
    '''
    from OpenHHEA.train import HHEATrainer, SimpleHHEALoss
    from OpenHHEA.reasoning import Simple_HHEA, ReasoningEmbeddingBased
    from OpenHHEA.reasoning.utils import get_noise_embeddings
    ### training
    MODEL_PATH = os.path.join(configs.model_save_dir, "model.pth")
    model = Simple_HHEA(
        time_span=1 + 27*13,
        ent_name_emb=get_noise_embeddings(processor.name_embeddings, noise_ratio=configs.name_noise_ratio),
        ent_time_emb=processor.time_embeddings,
        ent_struct_emb=processor.struct_embeddings,
        use_structure=configs.use_structure,
        use_time=configs.use_time,
        emb_size=configs.emb_size,
        structure_size=configs.structure_size,
        time_size=configs.time_size,
        device=configs.device
    )
    trainer = HHEATrainer(
        config=configs, 
        loss_fn=SimpleHHEALoss(gamma=configs.gamma),
        model_save_path=MODEL_PATH
    )
    trainer.train(
        model=model,
        train_alignments=loader.sup_pairs,
        dev_alignments=loader.ref_pairs,
        ent_num=loader.get_num_of_entity()
    )
    ### reasoning
    reasoning = ReasoningEmbeddingBased(
        config=configs,
        dataloader=loader,
        dataprocessor=processor,
        model=model,
        model_path=MODEL_PATH
    )
    results = reasoning.run(save_result_path=result_path)

    return results


def predefined_reasoning(
    configs:OpenHHEA.configs.model_configs.SimpleHHEAConfig, 
    loader:OpenHHEA.KGDataLoader, 
    processor:OpenHHEA.KGDataProcessor,
    result_path:str
) -> SimpleHHEAResult:
    '''
    Example code for entity alignment using the provided predefined reasoning workflow.
    '''
    ### run SimpleHHEA
    pipeline = OpenHHEA.init_model_pipeline(
        method_type="SimpleHHEA",
        config=configs,
        dataloader=loader,
        dataprocessor=processor
    )
    results = pipeline.run(save_result_path=result_path)
    return results




if __name__ == "__main__":
    '''
    Set configurations for model and data processing
    '''
    configs = OpenHHEA.get_model_config("SimpleHHEA")
    configs.set_config(
        device=0,
        name_noise_ratio=0.1,
        emb_size=64,
        structure_size=8,
        time_size=8,

        use_struct=True,
        use_time=True,

        gamma=1.0,
        lr=0.01,
        weight_decay=0.001,
        epochs=500,
        model_save_dir="SimpleHHEA_trained_models"
    )


    '''
    Load knowledge graph data
    '''
    loader = OpenHHEA.KGDataLoader(data_dir="data/icews_wiki")


    '''
    Process data to get embeddings
    '''
    ###### process configs of model
    name_process_configs = BertProcessConfigs()
    ### Or you can customize the data processing configs like this
    # name_process_configs = ProcessConfigs(method_type="bert", bert_model_path="albert-base-v2", device=0)
    struct_process_configs = FualignProcessConfigs(q=0.7)

    processor = OpenHHEA.KGDataProcessor(
        dataloader=loader,
        name_process_configs=name_process_configs,
        struct_process_configs=struct_process_configs,
        image_process_configs=None,
        entity_process_configs=None
    )


    '''
    Entity alignment
    '''
    RESULT_PATH = "results/results_SimpleHHEA_on_icews_wiki.txt"
    ### customized
    results = customized_reasoning(configs, loader, processor, RESULT_PATH)
    ### predifined
    results = predefined_reasoning(configs, loader, processor, RESULT_PATH)
