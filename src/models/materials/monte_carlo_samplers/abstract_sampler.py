from abc import ABC, abstractmethod
from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from src.constants import OutputsType
from src.utils.base import BaseModule
from src.utils.sampling_utils import AbstractSampler, Samples


class AbstractMonteCarloSampler(BaseModule, ABC):
    @dataclass
    class Config(BaseModule.Config):
        detached_strategy: bool = True
        num_samples: int = 1
        suffix: str = "ray-samples"

    cfg: Config

    def generate_samples(
        self,
        outputs: OutputsType,
        sampler: AbstractSampler,
    ) -> Samples:
        ret = self._generate_samples_impl(
            outputs=outputs,
            sampler=sampler,
        )
        if self.cfg.detached_strategy:
            return ret.detach()
        return ret

    @abstractmethod
    def _generate_samples_impl(
        self,
        outputs: OutputsType,
        sampler: AbstractSampler,
    ) -> Samples:
        pass

    def pdf(
        self,
        outputs: OutputsType,
        directions: Float[Tensor, "B S num_samples 3"],
    ) -> Float[Tensor, "B S num_samples 1"]:
        ret = self._pdf_impl(
            outputs=outputs,
            directions=directions,
        )
        if self.cfg.detached_strategy:
            return ret.detach()
        return ret

    @abstractmethod
    def _pdf_impl(
        self,
        outputs: OutputsType,
        directions: Float[Tensor, "B S num_samples 3"],
    ) -> Float[Tensor, "B S num_samples 1"]:
        pass
