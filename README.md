# HRC RL for Loco-Manip with the Unitree G1

# TODOS: 

* Reference and cite any AMO stuff needed to be cited
* Combine the movable object in this file
* Add grasping to the arms
* Train on grasping? Pushing button? how do you generalize it
* John? Use distilled AMO for locomotion
* Figure CPU vs GPU performance 

## Installation instructions

1. Ensure you have conda installed
2. Run the following commands, there may be some issues with graphics drivers

```bash
conda create -n amo python=3.8
conda activate amo
pip install -r requirements.txt
```

## Basic usage instructions

For simple teleop of the robot

```bash
python play_amo.py
```

## Basic RL Training Info

To train use the train.py file there are a few arguments that can be used

--total_timesteps (timesteps to train)
--learning_rate (learning rate)
--use_wandb (for disabling wandb logging)
--headless (for enabling visuals of sim)

There are some others as well but these are the main ones needed

```bash
Train:
python train.py --total_timesteps 1000000 --learning_rate 5e-4 --use_wandb true --headless true --n_envs 2

Run:
python eval.py --model_path checkpoints/final_model --num_episodes 5 --render true
```

## Eval info

``` bash
python eval.py \
  --model_path checkpoints/final_model \
  --num_episodes 5 \
  --render True \
  --use_wandb True
```

## Configuration

Look at config.py for more info

## Reward function

Currently consists of solely distance from goal and an action penalty
