# hilserl-torch

A PyTorch implementation of Hilserl - a distributed reinforcement learning framework for robotic manipulation tasks, particularly designed for Franka Panda robot arm control in simulation environments.

## Features

- 🤖 **SAC Algorithm**: Deep reinforcement learning based on Soft Actor-Critic (SAC)
- 🎥 **Visual Observations**: ResNet encoders for image processing with pretrained model support
- 🌐 **Distributed Training**: Learner-actor architecture supporting multi-process parallel training
- 👤 **Expert Intervention**: Human-in-the-loop training with keyboard intervention (press SPACE) to use expert policy
- 📊 **Demonstration Learning**: Support for learning from demonstration data (RLPD - Reinforcement Learning from Pre-collected Demonstrations)
- 🎮 **Simulation Environment**: MuJoCo-based Franka Panda robot manipulation simulation
- 📈 **Experiment Tracking**: Integrated WandB for experiment logging and visualization
- 💾 **Efficient Storage**: Memory-efficient replay buffer implementation

## Requirements

- Python 3.8
- CUDA 12.4+ (recommended for GPU acceleration)
- PyTorch 2.4.1+
- MuJoCo 2.3.7+
- See `environment.yml` for full dependency list

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd hilserl-torch
```

### 2. Create Conda environment

```bash
conda env create -f environment.yml
conda activate hilserl-torch
```

### 3. Install the package

```bash
pip install -e .
```

## Quick Start

### Basic Training Workflow

The project uses a distributed training architecture. You need to start both learner and actor processes separately.

#### 1. Start Learner (Training Process)

```bash
python examples/train_rlpd_sim.py \
    --learner \
    --seed 0 \
    --batch_size 256 \
    --training_starts 1000 \
    --critic_actor_ratio 4 \
    --encoder_type resnet18-pretrained \
    --checkpoint_period 5000 \
    --checkpoint_path actor_ckpt/
```

#### 2. Start Actor (Data Collection Process)

In another terminal:

```bash
python examples/train_rlpd_sim.py \
    --actor \
    --render \
    --seed 0 \
    --random_steps 1000 \
    --encoder_type resnet18-pretrained \
    --ip localhost
```

### Training with Demonstration Data

If you have demonstration data, you can load it when starting the learner:

```bash
python examples/train_rlpd_sim.py \
    --learner \
    --demo_path path/to/demo_data.pkl \
    # ... other arguments
```

### Expert Intervention Training

Specify the expert model path when starting the actor. During training, press SPACE to switch to expert policy:

```bash
python examples/train_rlpd_sim.py \
    --actor \
    --expert_agent_path path/to/expert_model.pt \
    --render \
    # ... other arguments
```

## Project Structure

```
hilserl-torch/
├── examples/                    # Examples and experiment scripts
│   ├── train_rlpd_sim.py       # Main training script
│   └── experiments/             # Experiment configurations
│       └── pick_cube_sim/      # Pick cube task configuration
├── franka_sim/                 # Franka robot simulation environment
│   ├── envs/                   # Gymnasium environment implementations
│   │   └── panda_pick_gym_env.py
│   └── controllers/            # Controller implementations
├── serl_launcher/              # Core training framework
│   ├── agents/                 # Reinforcement learning algorithms
│   │   └── continuous/
│   │       └── sac.py          # SAC algorithm implementation
│   ├── data/                   # Data storage and replay buffers
│   ├── networks/               # Neural network architectures
│   ├── vision/                 # Vision encoders
│   ├── wrappers/               # Environment wrappers
│   └── utils/                  # Utility functions
├── serl_robot_infra/           # Robot infrastructure (real robot support)
└── setup.py                    # Package installation configuration
```

## Key Parameters

### Learner Parameters

- `--learner`: Enable learner mode
- `--batch_size`: Training batch size (default: 512)
- `--training_starts`: Number of samples to collect before training starts (default: 500)
- `--critic_actor_ratio`: Ratio of critic to actor updates (default: 4)
- `--encoder_type`: Encoder type, options: `resnet18-pretrained`, `resnet`, `small`, etc.
- `--checkpoint_period`: Period to save checkpoints (0 means no saving)
- `--checkpoint_path`: Path to save checkpoints
- `--demo_path`: Path to demonstration data (optional)
- `--preload_rlds_path`: Path to preload RLDS data (optional)

### Actor Parameters

- `--actor`: Enable actor mode
- `--render`: Enable environment rendering
- `--random_steps`: Number of random exploration steps (default: 500)
- `--ip`: IP address of the learner server (default: localhost)
- `--expert_agent_path`: Path to expert model (for intervention training)
- `--max_steps`: Maximum training steps (default: 1000000)

### Common Parameters

- `--env`: Environment name (default: PandaVVS-v0)
- `--seed`: Random seed (default: 42)
- `--exp_name`: WandB experiment name
- `--debug`: Debug mode (disables WandB logging)
- `--use_amp`: Use mixed precision training (default: True)

## Environment Configuration

The project supports various environment configurations. You can customize them in `examples/experiments/config.py`:

- `single-arm-fixed-gripper`: Single arm with fixed gripper
- `single-arm-learned-gripper`: Single arm with learned gripper control
- `dual-arm-fixed-gripper`: Dual arm with fixed gripper
- `dual-arm-learned-gripper`: Dual arm with learned gripper control

## Training Tips

1. **Distributed Training**: Learner and Actor can run on different machines by specifying the correct IP address
2. **Mixed Precision Training**: AMP is enabled by default, which can speed up training and save GPU memory
3. **Demonstration Data**: Using high-quality demonstration data can significantly improve training performance
4. **Expert Intervention**: Using expert intervention in early training stages can help collect high-quality data
5. **Checkpoints**: Save checkpoints regularly to avoid data loss from training interruptions

## Development

### Adding New Environments

1. Create a new environment class in `franka_sim/envs/`
2. Inherit from `MujocoGymEnv` and implement necessary interfaces
3. Add corresponding configuration files in `examples/experiments/`

### Adding New Algorithms

1. Implement a new algorithm class in `serl_launcher/agents/`
2. Implement `sample_actions` and `update` methods
3. Integrate the new algorithm in the training script

## License

MIT License - see [LICENSE](LICENSE) file for details

## Contributing

Issues and Pull Requests are welcome!

## Acknowledgments

This project is based on the [hil-serl](https://github.com/rail-berkeley/hil-serl) framework, reimplemented in PyTorch.

## Contact

For questions or suggestions, please open an Issue.
