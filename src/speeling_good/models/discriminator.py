"""PyTorch architecture for spelling-error BIO token classification."""

from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput

from speeling_good.dataloaders.annotation_to_bio import ID2LABEL, LABEL2ID


class SpeelingGoodDescriminator(nn.Module):
    """Wrap a token-classification backbone with spelling-error BIO labels.

    With no backbone configuration, load pretrained weights from ``model_name``.
    Supply a configuration to construct the architecture without downloading
    weights, for example before restoring a saved state dictionary.
    """

    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        backbone_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if backbone_config is None:
            self.backbone = AutoModelForTokenClassification.from_pretrained(
                model_name,
                num_labels=len(LABEL2ID),
                id2label=ID2LABEL,
                label2id=LABEL2ID,
                attn_implementation="sdpa",
            )
        else:
            config_values = dict(backbone_config)
            model_type = config_values.pop("model_type")
            config = AutoConfig.for_model(model_type, **config_values)
            self.backbone = AutoModelForTokenClassification.from_config(
                config, attn_implementation="sdpa"
            )
        self.backbone.train()

    def forward(self, **inputs: torch.Tensor) -> TokenClassifierOutput:
        return self.backbone(**inputs)
