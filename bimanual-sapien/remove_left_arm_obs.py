import gymnasium as gym
import numpy as np

class RemoveLeftArmObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.left_arm_indices = list(range(0, 12, 2)) + list(range(12, 28))
        
        if "state" in self.observation_space.spaces:
            orig_space = self.observation_space.spaces["state"]
            new_dim = orig_space.shape[0] - len(self.left_arm_indices)
            self.observation_space.spaces["state"] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(new_dim,),
                dtype=orig_space.dtype
            )

    def observation(self, obs):
        if "state" in obs:
            state = obs["state"]
            mask = np.ones(state.shape[-1], dtype=bool)
            mask[self.left_arm_indices] = False
            obs["state"] = state[..., mask]
        return obs
