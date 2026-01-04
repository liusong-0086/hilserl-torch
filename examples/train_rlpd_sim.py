#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")

# Suppress asyncio socket.send() warnings
import logging
logging.getLogger('asyncio').setLevel(logging.ERROR)

import time
import numpy as np
import tqdm
from absl import app, flags
import os

from typing import Optional
import pickle as pkl
import gymnasium as gym
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics

import torch

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.utils.train_utils import concat_batches

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from serl_launcher.utils.launcher import (
    make_trainer_config,
    make_wandb_logger,
    make_sac_pixel_agent
)
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper

import franka_sim
import threading
from pynput import keyboard

FLAGS = flags.FLAGS

flags.DEFINE_string("env", "PandaPickCubeVision-v0", "Name of environment.")
flags.DEFINE_string("agent", "drq", "Name of agent.")
flags.DEFINE_string("exp_name", None, "Name of the experiment for wandb logging.")
flags.DEFINE_integer("max_traj_length", 100, "Maximum length of trajectory.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_bool("save_model", False, "Whether to save model.")
flags.DEFINE_integer("batch_size", 512, "Batch size.")
flags.DEFINE_integer("critic_actor_ratio", 4, "critic to actor update ratio.")

flags.DEFINE_integer("max_steps", 1000000, "Maximum number of training steps.")
flags.DEFINE_integer("replay_buffer_capacity", 200000, "Replay buffer capacity.")

flags.DEFINE_integer("random_steps", 500, "Sample random actions for this many steps.")
flags.DEFINE_integer("training_starts", 500, "Training starts after this step.")
flags.DEFINE_integer("steps_per_update", 30, "Number of steps per update the server.")

flags.DEFINE_integer("log_period", 10, "Logging period.")
flags.DEFINE_integer("eval_period", 2000, "Evaluation period.")
flags.DEFINE_integer("eval_n_trajs", 5, "Number of trajectories for evaluation.")

# flag to indicate if this is a learner or an actor
flags.DEFINE_boolean("learner", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("actor", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("render", False, "Render the environment.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
# "small" is a 4 layer convnet, "resnet" and "mobilenet" are frozen with pretrained weights
flags.DEFINE_string("encoder_type", "resnet18-pretrained", "Encoder type.")
flags.DEFINE_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_integer("checkpoint_period", 0, "Period to save checkpoints.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_string("expert_agent_path", None, "Path to the expert agent.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging

flags.DEFINE_boolean("use_amp", True, "Use mixed precision training (AMP) for faster training.")

flags.DEFINE_string("log_rlds_path", None, "Path to save RLDS logs.")
flags.DEFINE_string("preload_rlds_path", None, "Path to preload RLDS data.")


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def state_dict_to_numpy(state_dict, skip_optimizer=True):
    """Recursively convert a nested state dict to numpy arrays for network publishing.
    
    Args:
        state_dict: The state dict to convert
        skip_optimizer: If True, skip optimizer-related keys (they contain lists that can't be converted)
    """
    result = {}
    for k, v in state_dict.items():
        # Skip optimizer states - they have complex structures with lists
        if skip_optimizer and "optimizer" in k:
            continue
        # Skip config
        if k == "config":
            continue
            
        if isinstance(v, torch.Tensor):
            result[k] = v.detach().cpu().numpy()
        elif isinstance(v, dict):
            result[k] = state_dict_to_numpy(v, skip_optimizer=False)
        else:
            # Skip non-tensor items
            continue
    return result


def numpy_to_state_dict(params, device):
    """Recursively convert numpy arrays back to torch tensors."""
    result = {}
    for k, v in params.items():
        if isinstance(v, np.ndarray):
            result[k] = torch.as_tensor(v, device=device)
        elif isinstance(v, dict):
            result[k] = numpy_to_state_dict(v, device)
        else:
            result[k] = v
    return result


##############################################################################

class KeyboardInterventionMonitor:
    def __init__(self, intervention_key=keyboard.Key.space):
        self.intervention_key = intervention_key
        self.space_pressed = False
        self._key_lock = threading.Lock()
        
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self.listener.start()
    
    def _on_press(self, key):
        with self._key_lock:
            if key == self.intervention_key:
                self.space_pressed = True
    
    def _on_release(self, key):
        with self._key_lock:
            if key == self.intervention_key:
                self.space_pressed = False
    
    def is_pressed(self):
        with self._key_lock:
            return self.space_pressed
    
    def stop(self):
        self.listener.stop()


def actor(
    agent: SACAgent, 
    data_store, 
    intvn_data_store, 
    env, 
    expert_agent: Optional[SACAgent] = None,
    device: str = "cuda"
):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    agent.eval()
    # Create datastore dict with both buffers
    datastore_dict = {
        "actor_env": data_store,
        "actor_env_intvn": intvn_data_store,
    }
    
    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_stores=datastore_dict,
        wait_for_server=True,
    )

    # Function to update the agent with new params
    def update_params(params):
        """Update agent parameters from server"""
        state_dict = numpy_to_state_dict(params, device)
        # Use strict=False because proprio_encoder may be lazily initialized
        # at different times on actor vs learner
        agent.load_state_dict(state_dict, strict=False)

    client.recv_network_callback(update_params)

    intervention_monitor = None
    if expert_agent is not None:
        intervention_monitor = KeyboardInterventionMonitor()
        print_green("Expert intervention enabled. Press SPACE to use expert policy.")

    obs, _ = env.reset()
    done = False

    # training loop
    timer = Timer()
    running_return = 0.0
    intervention_count = 0
    intervention_steps = 0
    already_intervened = False

    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True):
        timer.tick("total")

        with timer.context("sample_actions"):
            intervened = False
            if step < FLAGS.random_steps:
                actions = env.action_space.sample()
            else:
                if intervention_monitor is not None and intervention_monitor.is_pressed():
                    with torch.no_grad():
                        obs_tensor = {
                            k: torch.as_tensor(v, device=device) 
                            for k, v in obs.items()
                        }
                        actions = expert_agent.sample_actions(
                            observations=obs_tensor,
                            argmax=True,
                        )
                    actions = actions.cpu().numpy()
                    intervened = True
                else:
                    with torch.no_grad():
                        obs_tensor = {
                            k: torch.as_tensor(v, device=device) 
                            for k, v in obs.items()
                        }
                        actions = agent.sample_actions(
                            observations=obs_tensor,
                            argmax=False,
                        )
                    actions = actions.cpu().numpy()

        # Step environment
        with timer.context("step_env"):
            next_obs, reward, done, truncated, info = env.step(actions)
            reward = np.asarray(reward, dtype=np.float32)
            
            # Track intervention statistics
            if intervened:
                intervention_steps += 1
                if not already_intervened:
                    intervention_count += 1
                already_intervened = True
            else:
                already_intervened = False
            
            running_return += reward

            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
                dones=done or truncated,
            )
            
            # All data goes into replay buffer
            data_store.insert(transition)
            
            # Intervention data additionally goes into intervention buffer
            if already_intervened:
                intvn_data_store.insert(transition)

            obs = next_obs
            if done or truncated:
                # Add intervention statistics to episode info
                if "episode" in info:
                    info["episode"]["intervention_count"] = intervention_count
                    info["episode"]["intervention_steps"] = intervention_steps
                
                stats = {"environment": info}
                client.request("send-stats", stats)
                running_return = 0.0
                intervention_count = 0
                intervention_steps = 0
                already_intervened = False
                client.update()
                obs, _ = env.reset()

        timer.tock("total")

        if step % FLAGS.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)


##############################################################################


def learner(
    agent: SACAgent,
    replay_buffer: MemoryEfficientReplayBufferDataStore,
    demo_buffer: Optional[MemoryEfficientReplayBufferDataStore] = None,
    device: str = "cuda",
):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    agent.train()
    # set up wandb and logging
    wandb_logger = make_wandb_logger(
        project="hilserl-torch",
        description=FLAGS.exp_name or FLAGS.env,
        debug=FLAGS.debug,
    )

    # To track the step in the training loop
    update_steps = 0

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=update_steps)
        return {}  # not expecting a response

    # Create server
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    # Register intervention buffer for collecting online intervention data
    if demo_buffer is not None:
        server.register_data_store("actor_env_intvn", demo_buffer)
    server.start(threaded=True)

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(
        total=FLAGS.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < FLAGS.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()

    # send the initial network to the actor
    server.publish_network(state_dict_to_numpy(agent.state_dict()))
    print_green("sent initial network to actor")

    # 50/50 sampling from RLPD, half from demo and half from online experience if
    # demo_buffer is provided
    if demo_buffer is None or len(demo_buffer) == 0:
        single_buffer_batch_size = FLAGS.batch_size
        demo_iterator = None
    else:
        single_buffer_batch_size = FLAGS.batch_size // 2
        demo_iterator = demo_buffer.get_iterator(
            sample_args={
                "batch_size": single_buffer_batch_size,
                "pack_obs_and_next_obs": True,
            },
            device=device,
        )

    # create replay buffer iterator
    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": single_buffer_batch_size,
            "pack_obs_and_next_obs": True,
        },
        device=device,
    )

    # wait till the replay buffer is filled with enough data
    timer = Timer()

    # show replay buffer progress bar during training
    pbar = tqdm.tqdm(
        total=FLAGS.replay_buffer_capacity,
        initial=len(replay_buffer),
        desc="replay buffer",
    )

    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True, desc="learner"):
        # run n-1 critic updates and 1 critic + actor update.
        # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU
        for critic_step in range(FLAGS.critic_actor_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)

                # we will concatenate the demo data with the online data
                # if demo_buffer is provided
                if demo_iterator is not None:
                    demo_batch = next(demo_iterator)
                    batch = concat_batches(batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                agent.update(batch, networks_to_update=frozenset({"critic"}))

        with timer.context("train"):
            batch = next(replay_iterator)

            # we will concatenate the demo data with the online data
            # if demo_buffer is provided
            if demo_iterator is not None:
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)
            update_info = agent.update(
                batch, 
                networks_to_update=frozenset({"actor", "critic", "temperature"})
            )

        # publish the updated network
        if step > 0 and step % (FLAGS.steps_per_update) == 0:
            torch.cuda.synchronize()
            with torch.no_grad():
                state_dict = agent.state_dict()
                numpy_params = state_dict_to_numpy(state_dict)
            server.publish_network(numpy_params)
            del state_dict, numpy_params
            torch.cuda.empty_cache()

        if update_steps % FLAGS.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=update_steps)
            wandb_logger.log({"timer": timer.get_average_times()}, step=update_steps)

        if FLAGS.checkpoint_period and update_steps > 0 and update_steps % FLAGS.checkpoint_period == 0:
            assert FLAGS.checkpoint_path is not None
            os.makedirs(FLAGS.checkpoint_path, exist_ok=True)
            checkpoint_file = os.path.join(
                FLAGS.checkpoint_path, f"checkpoint_{update_steps}.pt"
            )

            with torch.no_grad():
                torch.save({
                    'step': update_steps,
                    'model_state_dict': agent.state_dict(),
                }, checkpoint_file)
            print_green(f"Saved checkpoint to {checkpoint_file}")
            torch.cuda.empty_cache()

        pbar.update(len(replay_buffer) - pbar.n)  # update replay buffer bar
        update_steps += 1


##############################################################################


def main(_):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_green(f"Using device: {device}")
    
    # Set seeds for reproducibility
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(FLAGS.seed)

    if FLAGS.render:
        env = gym.make(FLAGS.env, render_mode="human")
    else:
        env = gym.make(FLAGS.env, render_mode="rgb_array")
    
    env = SERLObsWrapper(env)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    env = RecordEpisodeStatistics(env)

    image_keys = [key for key in env.observation_space.keys() if key != "state"]

    agent: SACAgent = make_sac_pixel_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=image_keys,
        encoder_type=FLAGS.encoder_type,
        discount=0.97
    )

    agent = agent.to(device)

    if FLAGS.learner:
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=FLAGS.replay_buffer_capacity,
            image_keys=image_keys,
            include_grasp_penalty=False,
            device="cpu",  # Store buffer on CPU to save GPU memory
        )
        
        # Create intervention buffer (used for both preloaded demos and online interventions)
        intvn_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=FLAGS.replay_buffer_capacity,
            image_keys=image_keys,
            include_grasp_penalty=False,
            device="cpu",
        )

        print_green("replay buffer created")
        print_green(f"replay_buffer size: {len(replay_buffer)}")
        print_green(f"intervention buffer size: {len(intvn_buffer)}")

        # if demo data is provided, load it into the intervention buffer
        if FLAGS.demo_path or FLAGS.preload_rlds_path:
            if FLAGS.demo_path:
                # Check if the file exists
                if not os.path.exists(FLAGS.demo_path):
                    raise FileNotFoundError(f"File {FLAGS.demo_path} not found")

                with open(FLAGS.demo_path, "rb") as f:
                    trajs = pkl.load(f)
                    for traj in trajs:
                        intvn_buffer.insert(traj)

            print_green(f"Loaded demo data. Intervention buffer size: {len(intvn_buffer)}")

        # learner loop (always use intvn_buffer for 50/50 sampling with online interventions)
        print_green("starting learner loop")
        learner(
            agent,
            replay_buffer,
            demo_buffer=intvn_buffer,  # Use intervention buffer
            device=device,
        )

    elif FLAGS.actor:
        # Policy in the loop
        expert_agent: Optional[SACAgent] = None
        if FLAGS.expert_agent_path is not None:
            expert_agent: SACAgent = make_sac_pixel_agent(
                seed=FLAGS.seed,
                sample_obs=env.observation_space.sample(),
                sample_action=env.action_space.sample(),
                image_keys=image_keys,
                encoder_type=FLAGS.encoder_type,
                discount=0.97,
                device=device,
                use_amp=False,  # Expert agent doesn't need AMP
            )

            expert_agent = expert_agent.to(device)
            
            # Load expert checkpoint
            checkpoint = torch.load(
                FLAGS.expert_agent_path, 
                map_location=device
            )
            if 'model_state_dict' in checkpoint:
                expert_agent.load_state_dict(checkpoint['model_state_dict'])
            else:
                expert_agent.load_state_dict(checkpoint)
            expert_agent.eval()
            print_green(f"Loaded expert agent from {FLAGS.expert_agent_path}")

        data_store = QueuedDataStore(50000)  # replay buffer data store
        intvn_data_store = QueuedDataStore(50000)  # intervention buffer data store

        # actor loop
        print_green("starting actor loop")
        actor(
            agent, 
            data_store, 
            intvn_data_store, 
            env, 
            expert_agent=expert_agent,
            device=device
        )

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)