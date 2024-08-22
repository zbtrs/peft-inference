from transformers import LlamaForCausalLM, LlamaModel
from typing import List, Optional, Tuple, Union
import torch

class LlamaSubmodel():
    def __init__(self, top_model: LlamaForCausalLM):
        self.top_model = top_model
        num_layers = len(self.top_model.model.layers)
        mid_index = num_layers // 2
        self.layers_part1 = self.top_model.model.layers[:mid_index]
        self.layers_part2 = self.top_model.model.layers[mid_index:]

    def forward(self,
                input_ids: torch.LongTensor = None,
                ):
