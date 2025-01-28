
        
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Any, List, Union
import numpy as np

from accelerate import Accelerator


class HFEncoder:
    def __init__(self, cfg) -> None:

        self.model_name = cfg.model.model_name
        self.accelerator = Accelerator()  # Initialize Accelerate
        self.device = self.accelerator.device
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model_configs, self.tokenizer_configs = cfg.model, cfg.tokenizer
        self.model, self.tokenizer = self.load_model()

        self.dataset_configs = cfg.dataset

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Prepare model and tokenizer for distributed setup
        self.model, self.tokenizer = self.accelerator.prepare(self.model, self.tokenizer)

    def load_model(self) -> Union[AutoModelForCausalLM, AutoTokenizer]:
        """Loads a HuggingFace model and tokenizer."""
        torch.cuda.empty_cache()
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, **self.tokenizer_configs.kwargs)
        model = AutoModelForCausalLM.from_pretrained(self.model_name, **self.model_configs.kwargs, device_map="auto")
        return model, tokenizer

    def encode(self, sentences: List[str]) -> torch.Tensor:
        """Process a batch of input sentences using the HuggingFace model and returns embeddings."""
        
        inputs = self.tokenizer(sentences, **self.tokenizer_configs.kwargs) 
        input_tokens = [self.tokenizer.convert_ids_to_tokens(input_ids) for input_ids in inputs['input_ids']]

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask, **self.model_configs.kwargs)
            hidden_states = outputs.hidden_states[-1] # Last layer hidden states: [batch_size, seq_len, hidden_dim]
            predicted_token_ids = outputs.logits.argmax(dim=-1)
            decoded_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in predicted_token_ids]

        expanded_mask = attention_mask.unsqueeze(-1)  # Shape: [batch, seq_len, 1]
        masked_embeddings = hidden_states * expanded_mask  # Zero out padding positions
        mean_pooled = masked_embeddings.sum(dim=1) / expanded_mask.sum(dim=1).clamp(min=1e-9)
    
        pad_left = attention_mask[:, 0].eq(0).all()
        if pad_left:
            batch_size = attention_mask.size(0)  # Get the batch size
            last_token_indices = torch.full((batch_size,), attention_mask.size(1) - 1, dtype=torch.long)
        else:
            last_token_indices = attention_mask.sum(dim=1) - 1
            
        last_token = hidden_states[torch.arange(hidden_states.size(0)), last_token_indices]
        
        self.accelerator.wait_for_everyone()
        torch.cuda.empty_cache()
    
        return input_tokens, mean_pooled, last_token
        
