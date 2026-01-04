from serl_robot_infra.robot_env.envs.wrappers import (
    Quat2EulerWrapper,
    GripperPenaltyWrapper,
)
from serl_robot_infra.robot_env.envs.relative_env import RelativeFrame
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper

from experiments.config import DefaultTrainingConfig
from franka_sim.envs.panda_pick_gym_env import PandaPickCubeGymEnv


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["front", "wrist"]
    classifier_keys = ["front", "wrist"]
    proprio_keys = ["tcp_pose", "tcp_vel", "gripper_pose"]
    buffer_period = 2000
    replay_buffer_capacity = 50000
    batch_size = 64
    random_steps = 0
    checkpoint_period = 5000
    steps_per_update = 50
    encoder_type = "resnet18-pretrained"
    setup_mode = "single-arm-learned-gripper"
    fake_env = False
    classifier = False

    def get_environment(self, fake_env=False, save_video=False, classifier=False, render_mode="rgb_array"):
        env = PandaPickCubeGymEnv(render_mode=render_mode, image_obs=True, time_limit=100.0, control_dt=0.1)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        env = GripperPenaltyWrapper(env, penalty=-0.02)
        return env