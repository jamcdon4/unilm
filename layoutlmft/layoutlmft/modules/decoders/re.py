import copy

import torch
from torch import nn
from torch.nn import CrossEntropyLoss


class BiaffineAttention(torch.nn.Module):
    """Implements a biaffine attention operator for binary relation classification.

    PyTorch implementation of the biaffine attention operator from "End-to-end neural relation
    extraction using deep biaffine attention" (https://arxiv.org/abs/1812.11275) which can be used
    as a classifier for binary relation classification.

    Args:
        in_features (int): The size of the feature dimension of the inputs.
        out_features (int): The size of the feature dimension of the output.

    Shape:
        - x_1: `(batch_size, *, in_features)` where `*` means any number of additional dimensisons.
        - x_2: `(batch_size, *, in_features)` where `*` means any number of additional dimensions.
        - Output: `(batch_size, *, out_features)` where `*` means any number of additional dimensions.


    Examples:
        >>> batch_size, in_features, out_features = 32, 100, 4
        >>> biaffine_attention = BiaffineAttention(in_features, out_features)
        >>> x_1 = torch.randn(batch_size, in_features)
        >>> x_2 = torch.randn(batch_size, in_features)
        >>> output = biaffine_attention(x_1, x_2)
        >>> print(output.size())
        torch.Size([32, 4])
    """

    def __init__(self, in_features, out_features):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.bilinear = torch.nn.Bilinear(in_features, in_features, out_features, bias=False)
        self.linear = torch.nn.Linear(2 * in_features, out_features, bias=True)

        self.reset_parameters()

    def forward(self, x_1, x_2):
        return self.bilinear(x_1, x_2) + self.linear(torch.cat((x_1, x_2), dim=-1))

    def reset_parameters(self):
        self.bilinear.reset_parameters()
        self.linear.reset_parameters()


class LayoutLMv2RelationExtractionDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.entity_emb = nn.Embedding(3, config.hidden_size, scale_grad_by_freq=True)
        projection = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.ffnn_head = copy.deepcopy(projection)
        self.ffnn_tail = copy.deepcopy(projection)
        self.rel_classifier = BiaffineAttention(config.hidden_size // 2, 2)
        self.loss_fct = CrossEntropyLoss()

    def get_predicted_relations(self, 
                                logits, 
                                relation_head_doc,
                                relation_tail_doc,
                                relation_label_doc,
                                entities_start_doc,
                                entities_label_doc,
                                entities_end_doc
        ):

        pred_relations = []
        for i, pred_label in enumerate(logits.argmax(-1)):
            if pred_label != 1:
                continue
            rel = {}
            rel["head_id"] = relation_head_doc[i]
            rel["head"] = (entities_start_doc[rel["head_id"]], entities_end_doc[rel["head_id"]])
            rel["head_type"] = entities_label_doc[rel["head_id"]]

            rel["tail_id"] = relation_tail_doc[i]
            rel["tail"] = (entities_start_doc[rel["tail_id"]], entities_end_doc[rel["tail_id"]])
            rel["tail_type"] = entities_label_doc[rel["tail_id"]]
            rel["type"] = 1
            pred_relations.append(rel)
        return pred_relations

    def forward(self, 
                hidden_states, 
                relation_head,
                relation_tail,
                relation_label,
                entities_start,
                entities_end,
                entities_label
        ):
        
        batch_size, max_n_words, context_dim = hidden_states.size()
        loss = 0
        all_pred_relations = []
        
        for doc in range(batch_size):
            
            relation_head_doc  = relation_head[doc] 
            relation_tail_doc  = relation_tail[doc]
            relation_label_doc = relation_label[doc]
            entities_start_doc = entities_start[doc]
            entities_label_doc = entities_label[doc]
            entities_end_doc = entities_end[doc]

            head_index = entities_start_doc[relation_head_doc]
            head_label = entities_label_doc[relation_head_doc]
            head_label_repr = self.entity_emb(head_label)

            tail_index = entities_start_doc[relation_tail_doc]
            tail_label = entities_label_doc[relation_tail_doc]
            tail_label_repr = self.entity_emb(tail_label)

            head_repr = torch.cat(
                (hidden_states[doc][head_index], head_label_repr),
                dim=-1,
            )
            tail_repr = torch.cat(
                (hidden_states[doc][tail_index], tail_label_repr),
                dim=-1,
            )
            heads = self.ffnn_head(head_repr)
            tails = self.ffnn_tail(tail_repr)
            logits = self.rel_classifier(heads, tails)
            loss += self.loss_fct(logits, relation_label_doc)
            
            pred_relations = self.get_predicted_relations(
                logits, 
                relation_head_doc,
                relation_tail_doc,
                relation_label_doc,
                entities_start_doc,
                entities_label_doc,
                entities_end_doc)
            
            all_pred_relations.append(pred_relations)

        return loss, all_pred_relations
