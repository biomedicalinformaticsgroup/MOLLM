import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Dict, List, Tuple, Any, Union, Optional
from functools import partial
from embed import LLMEmbed
from models import HFEncoder
from dataset import MDSUPDRS

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    dataset = MDSUPDRS(cfg.dataset)

    # Initialize a HuggingFace Encoder
    model = HFEncoder(cfg)
    embedder = LLMEmbed(dataset, batch_size=cfg.exp.batch_size)

    results = embedder(model)
    print(results)

if __name__ == "__main__":
    main()