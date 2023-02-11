from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from transformers.file_utils import ModelOutput


@dataclass
class ReOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.IntTensor]] = None
    attentions: Optional[Tuple[torch.IntTensor]] = None
    relation_head: Optional[torch.IntTensor] = None
    relation_tail: Optional[torch.IntTensor] = None
    relation_label: Optional[torch.IntTensor] = None
    entities_start: Optional[torch.IntTensor] = None
    entities_end: Optional[torch.IntTensor] = None
    entities_label: Optional[torch.IntTensor] = None
    pred_relations: Optional[Dict] = None
