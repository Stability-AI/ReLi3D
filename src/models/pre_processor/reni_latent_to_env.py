from dataclasses import dataclass, field

from src.constants import Names, OutputsType
from src.models.illumination.reni.env_map import RENIEnvMap
from src.utils.base import BaseModule


class ReniLatentToEnvProcessor(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        reni_env_config: dict = field(default_factory=dict)
        suffix: str = ""

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.reni_env = RENIEnvMap(RENIEnvMap.Config(**self.cfg.reni_env_config))

    def forward(self, outputs: OutputsType) -> OutputsType:
        env_map = self.reni_env(
            {
                Names.BATCH_SIZE: outputs[Names.BATCH_SIZE],
                Names.RENI_LATENT: outputs[
                    Names.RENI_LATENT.add_suffix(self.cfg.suffix)
                ].view(-1, self.reni_env.field.latent_dim, 3),
                Names.ILLUMINATION_ROTATION: outputs[
                    Names.ILLUMINATION_ROTATION.add_suffix(self.cfg.suffix)
                ].view(-1, 3, 3),
                Names.ILLUMINATION_STRENGTH: outputs[
                    Names.ILLUMINATION_STRENGTH.add_suffix(self.cfg.suffix)
                ].view(-1),
            }
            | (
                {
                    Names.ILLUMINATION_Z_ROTATION_RADS: outputs[
                        Names.ILLUMINATION_Z_ROTATION_RADS.add_suffix(self.cfg.suffix)
                    ].view(-1)
                }
                if Names.ILLUMINATION_Z_ROTATION_RADS.add_suffix(self.cfg.suffix)
                in outputs
                else {}
            )
        )
        return {Names.ENV_MAP: env_map[Names.RADIANCE.add_suffix("env")]}

    def produced_keys(self):
        return super().produced_keys().union({Names.ENV_MAP})

    def consumed_keys(self):
        return (
            super()
            .consumed_keys()
            .union(
                {
                    Names.RENI_LATENT.add_suffix(self.cfg.suffix),
                    Names.ILLUMINATION_ROTATION.add_suffix(self.cfg.suffix),
                    Names.ILLUMINATION_STRENGTH.add_suffix(self.cfg.suffix),
                    Names.ILLUMINATION_Z_ROTATION_RADS.add_suffix(self.cfg.suffix),
                }
            )
        )
