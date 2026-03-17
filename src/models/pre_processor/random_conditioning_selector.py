import random
from dataclasses import dataclass

import torch

from src.constants import Names, OutputsType
from src.utils.base import BaseModule
from src.utils.typing import List


class RandomViewElementConditioningSelector(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        min_condition_count: int = 1
        max_condition_count: int = 1
        training_only: bool = True

    cfg: Config

    inplace_module = True

    def find_all_condition_keys(self, outputs: OutputsType) -> List[str]:
        potential_condition_keys = [
            k for k in outputs if k.is_cond and not k.is_meta_data
        ]
        batch_view_size_per_key = {
            k: outputs[k].shape[:2] for k in potential_condition_keys
        }

        # Fine the most agreeable batch size and view size. Count the occurrences of each batch size and view size.
        batch_size_counts = {}
        view_size_counts = {}
        for k, (batch_size, view_size) in batch_view_size_per_key.items():
            batch_size_counts[batch_size] = batch_size_counts.get(batch_size, 0) + 1
            view_size_counts[view_size] = view_size_counts.get(view_size, 0) + 1
        most_agreeable_batch_size = max(batch_size_counts, key=batch_size_counts.get)
        most_agreeable_view_size = max(view_size_counts, key=view_size_counts.get)

        # Filter out the keys that do not have the most agreeable batch size and view size.
        return [
            k
            for k in potential_condition_keys
            if batch_view_size_per_key[k]
            == (most_agreeable_batch_size, most_agreeable_view_size)
        ]

    def forward(self, outputs: OutputsType) -> OutputsType:
        if self.cfg.training_only and not self.training:
            return {}
        condition_keys = self.find_all_condition_keys(outputs)
        example_condition = outputs[condition_keys[0]]
        assert (
            example_condition.ndim >= 2
        ), f"Conditioning must be at least 2D, got {example_condition.shape} for key {condition_keys[0]}"
        batch_size, view_size = example_condition.shape[:2]
        num_elements = random.randint(
            self.cfg.min_condition_count, self.cfg.max_condition_count
        )
        selected_batch_elements = torch.randperm(view_size)[:num_elements]
        ret = {}
        for condition_key in condition_keys:
            assert condition_key in outputs
            ret[condition_key] = outputs[condition_key][:, selected_batch_elements]

        ret[Names.VIEW_SIZE.cond] = selected_batch_elements.shape[0]
        return ret
