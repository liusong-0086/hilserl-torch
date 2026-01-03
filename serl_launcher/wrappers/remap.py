from typing import Any

import gymnasium as gym


def _tree_map_observation(fn, structure):
    """Apply fn to leaves of a nested structure (tuple, dict, or leaf)."""
    if isinstance(structure, tuple):
        return tuple(_tree_map_observation(fn, v) for v in structure)
    elif isinstance(structure, dict):
        return {k: _tree_map_observation(fn, v) for k, v in structure.items()}
    else:
        # Leaf node
        return fn(structure)


class RemapWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, new_structure: Any):
        """
        Remap a dictionary observation space to some other flat structure specified by keys.

        Params:
            env: Environment to wrap.
            new_structure: A tuple/dictionary/singleton where leaves are keys in the original observation space.
        """
        super().__init__(env)
        self.new_structure = new_structure

        if isinstance(new_structure, tuple):
            self.observation_space = gym.spaces.Tuple(
                [env.observation_space[v] for v in new_structure]
            )
        elif isinstance(new_structure, dict):
            self.observation_space = gym.spaces.Dict(
                {k: env.observation_space[v] for k, v in new_structure.items()}
            )
        elif isinstance(new_structure, str):
            self.observation_space = env.observation_space[new_structure]
        else:
            raise TypeError(f"Unsupported type {type(new_structure)}")

    def observation(self, observation):
        return _tree_map_observation(lambda x: observation[x], self.new_structure)
