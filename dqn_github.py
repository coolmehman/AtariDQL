# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
import os
import random
import time
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10.0, 8.0) # default plot size
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

import random

import tensorflow as tf
import datetime

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   Args
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@dataclass
class Args:
    #exp_name: str = os.path.basename(__file__)[: -len(".py")]
    exp_name: str = "AtariDQL"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "AtariDQN"
    """the wandb's project name"""
    #wandb_entity: str = "ericericks-Redondo Beach Unified School District"
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "PongNoFrameskip-v4"
    """the id of the environment"""
    #total_timesteps: int = 10000000
    total_timesteps: int = 10000
    #100k steps takes ~8-10 minutes (On GPU)
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    #buffer_size: int = 1000000
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 100
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    #learning_starts: int = 80000
    learning_starts: int = 500
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""
    heatmap_frequency: int = 10
    """the frequency of heatmap update"""
    heatmap_change_frequency : int = 100
    """the frequency of the comparison image for heatmap changing"""
    heatmap_upload_frequency: int = 100
    """the frequency of heatmap upload"""

ArgsDict = {
  "exp_name" : Args.exp_name,
  """the name of this experiment"""
  "seed" : Args.seed,
  """seed of the experiment"""
  "torch_deterministic" : Args.torch_deterministic,
  """if toggled, `torch.backends.cudnn.deterministic=False`"""
  "cuda" : Args.cuda,
  """if toggled, cuda will be enabled by default"""
  "track" : Args.track,
  """if toggled, this experiment will be tracked with Weights and Biases"""
  "wandb_project_name" : Args.wandb_project_name,
  """the wandb's project name"""
  "wandb_entity" : Args.wandb_entity,
  """the entity (team) of wandb's project"""
  "capture_video" : Args.capture_video,
  """whether to capture videos of the agent performances (check out `videos` folder)"""
  "save_model" : Args.save_model,
  """whether to save model into the `runs/{run_name}` folder"""
  "upload_model" : Args.upload_model,
  """whether to upload the saved model to huggingface"""
  "hf_entity" : Args.hf_entity,
  """the user or org name of the model repository from the Hugging Face Hub"""

  # Algorithm specific arguments
  "env_id" : Args.env_id,
  """the id of the environment"""
  #total_timesteps: int = 10000000
  "total_timesteps" : Args.total_timesteps,
  """total timesteps of the experiments"""
  "learning_rate" : Args.learning_rate,
  """the learning rate of the optimizer"""
  "num_envs" : Args.num_envs,
  """the number of parallel game environments"""
  #buffer_size: int = 1000000
  "buffer_size" : Args.buffer_size,
  """the replay memory buffer size"""
  "gamma" : Args.gamma,
  """the discount factor gamma"""
  "tau" : Args.tau,
  """the target network update rate"""
  "target_network_frequency" : Args.target_network_frequency,
  """the timesteps it takes to update the target network"""
  "batch_size" : Args.batch_size,
  """the batch size of sample from the reply memory"""
  "start_e" : Args.start_e,
  """the starting epsilon for exploration"""
  "end_e" : Args.end_e,
  """the ending epsilon for exploration"""
  "exploration_fraction" : Args.exploration_fraction,
  """the fraction of `total-timesteps` it takes from start-e to go end-e"""
  #learning_starts: int = 80000
  "learning_starts" : Args.learning_starts,
  """timestep to start learning"""
  "train_frequency" : Args.train_frequency,
  """the frequency of training"""
  "heatmap_frequency" : Args.heatmap_frequency,
  """the frequency of the heatmap update"""
  "heatmap_change_frequency" : Args.heatmap_change_frequency,
  """the frequency of the comparison image for heatmap changing"""
  "heatmap_upload_frequency" : Args.heatmap_upload_frequency
  #"""the frequency of the heatmap upload"""
}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   Functions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        return env

    return thunk

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def update_heatmap(curr_obs, comp_obs, q_network, dqn_heatmap):
  #Curr_obs comp_obs naming somewhat innaccurate, curr_obs is the obs which the decision that is being measured comes from
  #Should really add checks for obs's to see if they have that empty first layer (curr_obs[0] is actual data)
  device = torch.device("cuda" if torch.cuda.is_available() and Args.cuda else "cpu")

  #Scaling so that, accounting for how many times this
  # function is run, final heatmap values are visible 
  #heatmap_change_rate = 1.0
  heatmap_change_rate = 1000000.0 / (Args.total_timesteps/float(Args.heatmap_frequency))
  
  changemap = np.array([[0.0]*84]*84)

  scene_count, row_count, column_count = 0, 0, 0
  for scene in curr_obs:
    row_count = 0
    for row in scene:
      column_count = 0
      for pixel in row:
        #int(~~~) serves to cast uint8 to int to allow negatives
        if abs(int(pixel) - int(comp_obs[scene_count][row_count][column_count])) > 10:
          changemap[row_count][column_count] += 0.25
        column_count += 1
      row_count += 1
    scene_count += 1


  #Q values of current observation
  curr_q_values = q_network(torch.Tensor(np.array([curr_obs])).to(device))
  curr_q_values = np.array(curr_q_values.cpu().detach().numpy())[0]#Keeping only action probabilities
  #Q values of comparison observation
  comp_q_values = q_network(torch.Tensor(np.array([comp_obs])).to(device))
  comp_q_values = np.array(comp_q_values.cpu().detach().numpy())[0]#Keeping only action probabilities

  index_max_q = 0
  for action_index in range(0,5):
    if curr_q_values[action_index] > index_max_q:
      index_max_q = action_index

  action_q_change = abs(curr_q_values[index_max_q] - comp_q_values[index_max_q])

  row_count = 0
  for row in dqn_heatmap:
    column_count = 0
    for pixel in row:
      #Uses heatmap_change_rate to scale since action_q_change is usually in the hundreths or thousanths
      #print(pixel, (changemap[row_count][column_count] * action_q_change))
      dqn_heatmap[row_count][column_count] = pixel + ((changemap[row_count][column_count] * action_q_change) * heatmap_change_rate)
      column_count += 1
    row_count += 1

  # if(np.amax(dqn_heatmap) > 255.0):
  #   scaling_factor = 255.0 / np.amax(dqn_heatmap)
  #   #Scale heatmap values to 255, slightly depresses values that arent immediately relevant between the 
  #   # two images, so try to avoid pushing values past a bit more than 255 in the previous loop
  #   row_count = 0
  #   for row in dqn_heatmap:
  #     column_count = 0
  #     for pixel in row:
  #       dqn_heatmap[row_count][column_count] = pixel * scaling_factor
  #       column_count += 1
  #     row_count += 1

  return dqn_heatmap

def scale_arraymap(arraymap):
  scaling_factor = 255.0 / np.amax(arraymap)
  row_count = 0
  for row in arraymap:
    column_count = 0
    for pixel in row:
      arraymap[row_count][column_count] = pixel * scaling_factor
      column_count += 1
    row_count += 1
  #Might want to change this to arraymap.copy() to ensure value, not pointer, is returned
  return arraymap

def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
    capture_video: bool = True,
    log_heatmaps: bool = False
):
    if log_heatmaps:
      #single_eval_heatmaps will include heatmap and obs
      single_eval_heatmaps = []
      eval_heatmap = np.array([[0.0]*84]*84)

    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    model = Model(envs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = model(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        next_obs, _, _, _, infos = envs.step(actions)

        if log_heatmaps:
          #May want to mess with which obs's are used
          #Current idea is that by comparing to previous obs,
          # heatmap shows "moments" where decisions change
          update_heatmap(obs[0], next_obs[0], model, eval_heatmap)

          #Might be excessive .copy() with second one,
          # unsure if the pointer or value is returned
          eval_heatmap_obs = np.concatenate((scale_arraymap(eval_heatmap.copy()).copy(), next_obs[0][3].copy()), axis = 1)
          
          #These will be the individual frames when logged as video
          single_eval_heatmaps.append([eval_heatmap_obs.copy()]*3)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                  continue
                elif log_heatmaps:
                  #Log accumulated maps
                  run.log({"video":wandb.Video(np.asarray(single_eval_heatmaps), caption="eval_heatmap", fps=4, format="mp4")})
                  #Reset maps for next evaluation episode
                  single_eval_heatmaps = []
                  eval_heatmap = np.array([[0.0]*84]*84)
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   Main DQN Sequence
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#final_obs = np.array([[[[0]*84] * 84]*4])
#final_network;

if __name__ == "__main__":
    import stable_baselines3 as sb3

    rand_steps = np.array([0] * 10)
    for i in range(10):
      rand_steps[i] = random.randint(1, Args.total_timesteps - 1)

    rand_obs = []

    heatmap = np.array([[0.0]*84]*84)
    heatmaps = []

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1"
"""
        )
    #Cant figure out how to get this working
    #Using ArgsDict or directly accessing Args instead
    #args = tyro.cli(Args)
    assert Args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{Args.env_id}__{Args.exp_name}__{Args.seed}__{int(time.time())}"
    if Args.track:
        import wandb

        #WARNING When using several event log directories, please call `wandb.tensorboard.patch(root_logdir="...")` before `wandb.init`
        run = wandb.init(
            settings=wandb.Settings(start_method="thread"),
            project=Args.wandb_project_name,
            entity=Args.wandb_entity,
            sync_tensorboard=True,
            #config=vars(args),
            config=ArgsDict,
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(Args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(Args.seed)
    np.random.seed(Args.seed)
    torch.manual_seed(Args.seed)
    torch.backends.cudnn.deterministic = Args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and Args.cuda else "cpu")

    # env setup
    gym.make(Args.env_id)
    envs = gym.vector.SyncVectorEnv(
        [make_env(Args.env_id, Args.seed + i, i, Args.capture_video, run_name) for i in range(Args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=Args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        Args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=Args.seed)

    for global_step in range(Args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(Args.start_e, Args.end_e, Args.exploration_fraction * Args.total_timesteps, global_step)
        if random.random() < epsilon:
          actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
          q_values = q_network(torch.Tensor(obs).to(device))
          actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        if np.any(global_step == rand_steps):
          rand_obs.append(obs)

        # ALGO LOGIC: training.

        # if global_step == Args.learning_starts:
        #   comparison_obs = obs.copy()

        if global_step > Args.learning_starts:
            if global_step % Args.train_frequency == 0:
                data = rb.sample(Args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + Args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % Args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        Args.tau * q_network_param.data + (1.0 - Args.tau) * target_network_param.data
                    )
            
            # if global_step % Args.heatmap_frequency == 0:
            #   update_heatmap(obs[0], comparison_obs[0], q_network, heatmap)

            # if global_step % Args.heatmap_change_frequency == 0:
            #   comparison_obs = obs.copy()

            # if global_step % Args.heatmap_upload_frequency == 0:
            #   #AssertionError: size of input tensor and input format are different.
            #   # tensor shape: (84, 84), input_format: CHW
            #   #writer.add_image("heatmap", heatmap, global_step)

            #   #Append actual heatmap values rather than pointers to the final heatmap value
            #   heatmaps.append([heatmap.copy()]*3)

    # for i in range(5):
    #   if np.all(rand_obs[i] == np.array([[[0]*84]*84]*4)):
    #     rand_obs[i] = obs.copy()

    # final_q_network = q_network

    if Args.save_model:
        model_path = f"runs/{run_name}/{Args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        #from cleanrl_utils.evals.dqn_eval import evaluate
        #from dqn_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            Args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=0.05,
            capture_video= True,
            log_heatmaps = True
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        # Moved to evaluate()
        # if Args.track:
        #   for eval_heatmaps in heatmaps:
        #     run.log({"video":wandb.Video(np.asarray(eval_heatmaps), caption="heatmap", fps=4, format="mp4")})

        if Args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{Args.env_id}-{Args.exp_name}-seed{Args.seed}"
            repo_id = f"{Args.hf_entity}/{repo_name}" if Args.hf_entity else repo_name
            push_to_hub(Args, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
    if Args.track:
      run.finish()