from typing import Optional

from tensordict import TensorDict, TensorDictBase
from torchrl.envs import EnvBase
from torchrl.modules import ValueOperator
from torchrl.data import Spe
from env import NetworkEnvGym, TorchGraphNetworkxWrapper


class TorchNetworkEnv(EnvBase):
    def __init__(self, env):
        super().__init__()
        self._network_env = env
    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        pass

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        pass

    def _set_seed(self, seed: Optional[int]):
        pass


