"""
Create a reinforcement learning environment in Python following OpenAI Gym or Gymnasium standards (gym.Env) for the optimization problem described below. The environment should include all necessary components for training RL agents.

Problem Description:

A marketing company is planning to allocate resources between two projects: $X$ and $Y$. The total resource allocation for both projects cannot exceed 1000 units due to budget constraints. Project $X$ requires a minimum excess of 200 units over project $Y$ to ensure its success. Each unit of resource allocation costs \\$50 for project $X$ and \\$30 for project $Y$. Both projects can only be allocated whole numbers of resources due to the indivisible nature of these resources. Moreover, the maximum resource allocation for project X is 700 units and for project Y is 500 units due to operational constraints.\\n\\nGiven these conditions, what is the minimum cost (in dollars) required for resource allocation that satisfies all given constraints? Provide your answer rounded to the nearest dollar.

Environment Specifications:

State Representation: Define the state space that captures all relevant information about the current situation. The state should be a numerical vector or tuple that the agent can process.

Action Space: Design the action space considering the problem constraints. Specify whether actions are discrete or continuous, and define the valid action ranges. Account for any indivisible resource constraints.

Episode Reward (Objective): The primary reward signal computed at the end of each episode. This should align with the main optimization goal (e.g., minimize cost = negative cost reward). Structure rewards to encourage constraint satisfaction while optimizing the objective.

Step Penalty: Include a small negative reward for each time step to encourage efficient solutions and prevent infinite episodes.

Constraint Validation & Penalties:

Step Validation: Check constraints after each action. Apply immediate penalties for constraint violations during the episode.

Episode Validation: Apply final penalties if the terminal state violates any constraints, ensuring the agent learns to satisfy all requirements.

Termination Conditions: Define when an episode ends (e.g., when a valid solution is found, maximum steps reached, or constraints are irrecoverably violated).

Additional Requirements:

Implement the standard Gym methods: reset(), step(action), and optionally render()

Define observation and action spaces using gym.spaces

Include comprehensive constraint checking

Add clear documentation and comments

Ensure the environment is compatible with common RL libraries

Environment Review Guidelines:

Verify that all problem constraints are properly enforced

Ensure the reward structure guides the agent toward feasible optimal solutions

Test boundary conditions and edge cases

Validate that the state representation contains sufficient information for decision making

Generate complete, runnable Python code that implements this environment.

"""

from typing import Tuple
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.logger import configure
import numpy as np
import argparse

class ResourceAllocEnv(gym.Env):
    """Gymnasium-style environment for the resource allocation integer programming problem."""

    # Gymnasium uses 'render_modes' not 'render.modes'
    metadata = {"render_modes": ["human"]}

    def __init__(self, step_size: int = 10, max_steps: int = 200):
        super().__init__()

        # Problem constants
        self.total_budget = 1000
        self.x_min = 200
        self.x_max = 700
        self.y_min = 0
        self.y_max = 500
        self.x_minus_y_min = 200  # x - y >= 200

        # Costs
        self.cost_per_x = 50
        self.cost_per_y = 30

        # Action semantics: 0=inc_x, 1=dec_x, 2=inc_y, 3=dec_y, 4=submit
        self.action_space = spaces.Discrete(5)

        # Observation: current x, current y, remaining_budget = total - (x+y)
        # We'll represent observations as integers, but cast to float32 for networks
        low = np.array([self.x_min, self.y_min, 0], dtype=np.float32)
        high = np.array([self.x_max, self.y_max, self.total_budget], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Environment dynamics
        self.step_size = int(step_size)
        self.max_steps = int(max_steps)

        # defer RNG seeding to reset(seed=...)
        self.np_random = np.random.RandomState()

        # reward shaping coefficients
        # reward = k_cost * (prev_cost - curr_cost) - k_step - k_violation * total_violation
        self.k_cost = 10
        self.k_step = 200
        # penalty proportional to current violation magnitude (small)
        self.k_violation = 0.5
        # directional guidance: reward when total_violation decreases
        self.k_validation_dir = 100
        self.best_x = None
        self.best_y = None
        self.best_cost = None

    def seed(self, seed=None):
        # keep compatibility, though Gymnasium seeds via reset
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        # start with minimum allocations to make easier to find feasible solutions

        self.x = np.random.randint(20,70)*10
        self.y = np.random.randint(0, 50)*10

        self.current_step = 0
        self.done = False
        obs = self._get_obs()
        info = {"x": self.x, "y": self.y, "cost": int(self._cost(self.x, self.y))}
        return obs, info

    def _get_obs(self):
        remaining = self.total_budget - (self.x + self.y)
        return np.array([self.x, self.y, remaining], dtype=np.float32)

    def _check_constraints(self, x: int, y: int) -> Tuple[bool, dict]:
        """Return (is_feasible, violations)
        violations: dictionary with nonnegative numbers indicating amount of violation for each constraint
        """
        violations = {}
        violations['budget'] = max(0, x + y - self.total_budget)
        violations['x_minus_y'] = max(0, (self.x_minus_y_min - (x - y)))
        violations['x_upper'] = max(0, x - self.x_max)
        violations['x_lower'] = max(0, self.x_min - x)
        violations['y_upper'] = max(0, y - self.y_max)
        violations['y_lower'] = max(0, self.y_min - y)

        total_violation = sum(violations.values())
        return (total_violation == 0, violations)

    def _cost(self, x: int, y: int) -> int:
        return self.cost_per_x * x + self.cost_per_y * y

    def step(self, action: int):
        # Convert numpy action to Python int if needed
        if isinstance(action, (np.ndarray, np.generic)):
            action = int(np.asarray(action).item())

        assert self.action_space.contains(action), "Invalid action"
        if self.done:
            # if called after terminal, return no-op
            return self._get_obs(), 0.0, True, False, {"status": "already_done"}

        self.current_step += 1
        reward = 0.0
        info = {}
        terminated = False
        truncated = False

        # compute previous cost for cost-difference reward
        prev_cost = float(self._cost(self.x, self.y))
        # compute previous total violation for directional validation reward
        _, prev_violations = self._check_constraints(self.x, self.y)
        prev_total_violation = float(sum(prev_violations.values()))

        # Apply action
        if action == 0:  # increase x
            self.x = int(min(self.x + self.step_size, self.x_max))
        elif action == 1:  # decrease x
            self.x = int(max(self.x - self.step_size, self.x_min))
        elif action == 2:  # increase y
            self.y = int(min(self.y + self.step_size, self.y_max))
        elif action == 3:  # decrease y
            self.y = int(max(self.y - self.step_size, self.y_min))
        elif action == 4:  # submit
            feasible, violations = self._check_constraints(self.x, self.y)
            if feasible:
                cost = self._cost(self.x, self.y)
                reward = -float(cost)  # negative cost -> maximize reward leads to min cost
                info['status'] = 'feasible'
                info['cost'] = cost
                info['x'] = self.x
                info['y'] = self.y

            else:
                # Large penalty for infeasible solutions; also include magnitude of violations
                penalty = 100000
                reward = -penalty
                info['status'] = 'infeasible'
                info['violations'] = violations
            terminated = True

        # if not submit, compute cost-difference based reward + step penalty + directional validation bonus
        if not terminated:
            curr_cost = float(self._cost(self.x, self.y))
            feasible_step, violations_step = self._check_constraints(self.x, self.y)
            total_violation = float(sum(violations_step.values()))

            # reward is positive when cost decreased
            r_cost = self.k_cost * (prev_cost - curr_cost)
            r_step = -self.k_step
            # directional validation: positive when total_violation decreased
            r_validation = self.k_validation_dir * (prev_total_violation - total_violation)
            if r_validation < 0:
                r_validation = 2*r_validation  # amplify penalty for increasing violations

            reward = float(r_cost + r_step + r_validation)

            # attach violations and violation change to info for debugging/monitoring
            if total_violation > 0:
                info['violations'] = violations_step
            info['violation_change'] = float(prev_total_violation - total_violation)

        # automatic done if exceeded steps
        if not terminated and self.current_step >= self.max_steps:
            # force-submit at end of episode (time-limit truncation)
            feasible, violations = self._check_constraints(self.x, self.y)
            if feasible:
                cost = self._cost(self.x, self.y)
                reward = -float(cost) - 10.0  # extra small penalty for late finish
                info['status'] = 'feasible_end_of_steps'
                info['cost'] = cost
            else:
                penalty = 1e5
                reward = -penalty
                info['status'] = 'infeasible_end_of_steps'
                info['violations'] = violations
            truncated = True

        if terminated or truncated:
            self.done = True

        return self._get_obs(), reward, terminated, truncated, info

    def render(self, mode='human'):
        print(f"Step {self.current_step}: x={self.x}, y={self.y}, remaining={self.total_budget - (self.x+self.y)}")

# ---------------------------
# register the environment with Gymnasium
gym.envs.registration.register(
    id='ResourceAllocation-v0',
    entry_point=__name__ + ':ResourceAllocEnv',
)
# ---------------------------

def manual_mode(step_size: int = 10, max_steps: int = 200):
    """Interactive CLI to manually control the environment.

    Controls:
      - 0: inc_x (+step)
      - 1: dec_x (-step)
      - 2: inc_y (+step)
      - 3: dec_y (-step)
      - 4: submit (terminate)
      - r: reset episode
      - q: quit
    """
    env = ResourceAllocEnv(step_size=step_size, max_steps=max_steps)
    print("Manual mode started. Enter actions to interact with the env.")
    print("Actions: 0=inc_x, 1=dec_x, 2=inc_y, 3=dec_y, 4=submit, r=reset, q=quit")
    obs, info = env.reset()

    def print_state(prefix: str = ""):
        feasible, violations = env._check_constraints(env.x, env.y)
        remaining = env.total_budget - (env.x + env.y)
        cost = env._cost(env.x, env.y)
        print((prefix + " ").strip() + f"Step={env.current_step} | x={env.x}, y={env.y}, remaining={remaining}, cost={cost}, feasible={feasible}")
        if not feasible:
            print(f"  violations: {violations}")

    print_state("Initial")
    while True:
        try:
            cmd = input("action [0/1/2/3/4, r, q] > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting manual mode.")
            break

        if cmd == 'q':
            print("Quit.")
            break
        if cmd == 'r':
            obs, info = env.reset()
            print_state("Reset")
            continue

        action_map = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}
        if cmd not in action_map:
            print("Invalid input. Use 0/1/2/3/4, r, or q.")
            continue
        action = action_map[cmd]

        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        print_state()
        print(f"  reward={reward:.2f}, terminated={terminated}, truncated={truncated}, info={info}")
        if done:
            print("Episode finished. Press 'r' to reset or 'q' to quit.")

    env.close()

def train_and_eval(total_timesteps: int = 1_000_000):
    # Create a vectorized env compatible with Stable-Baselines3
    # Wrap each sub-environment with Monitor so episode returns are tracked
    def make_monitored_env():
        return Monitor(gym.make('ResourceAllocation-v0'))

    vec_env = make_vec_env(make_monitored_env, n_envs=4)
    # VecMonitor aggregates Monitor results across vectorized envs and exposes rollout metrics
    vec_env = VecMonitor(vec_env)

    # Create and train PPO with a stronger policy network (wider/deeper MLP)
    # For this low-dim problem a larger MLP often outperforms the default small one.
    # You can also try recurrent policies (RecurrentPPO from sb3-contrib) if temporal memory helps.
    policy_kwargs = dict(
        net_arch=[dict(pi=[64, 64], vf=[64, 64])]
    )
    # If you want a custom activation, import torch.nn and set activation_fn=torch.nn.ReLU
    model = PPO('MlpPolicy', vec_env, verbose=1, policy_kwargs=policy_kwargs)

    # Configure logger to print to stdout and write tensorboard logs
    new_logger = configure(folder="sb3_logs")
    model.set_logger(new_logger)

    model.learn(total_timesteps=total_timesteps, tb_log_name="PPO_ResourceAlloc")

    # Evaluate deterministically on a fresh single env using Gymnasium API
    eval_env = gym.make('ResourceAllocation-v0')
    obs, info = eval_env.reset()
    done = False
    total_reward = 0.0

    print("\n--- Evaluate learned policy ---")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
        total_reward += float(reward)
        eval_env.render()

    print(f"Episode done. Total reward: {total_reward}, total cost: {info.get('cost', 'N/A')}")
    if 'cost' in info:
        print(f"Final cost: ${info['cost']}")
    eval_env.close()

    return model


def evaluate_model(model, episodes: int = 20, deterministic: bool = True):
    """Run rollouts with a trained model and print summary results."""
    env = ResourceAllocEnv(step_size=10, max_steps=200)
    feasible = 0
    infeasible = 0
    best_cost = None
    best_xy = None
    rewards = []

    for i in range(episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, rew, term, trunc, info = env.step(int(action))
            done = bool(term or trunc)
            ep_reward += float(rew)
        status = info.get("status", "")
        cost = info.get("cost")
        x_final, y_final = env.x, env.y
        if status.startswith("feasible"):
            feasible += 1
            if cost is not None and (best_cost is None or cost < best_cost):
                best_cost = cost
                best_xy = (x_final, y_final)
        else:
            infeasible += 1
        print(f"Episode {i+1}: status={status}, x={x_final}, y={y_final}, cost={cost}, reward={ep_reward:.2f}")
        rewards.append(ep_reward)

    avg_reward = float(np.mean(rewards)) if rewards else 0.0
    print(f"Summary: feasible={feasible}/{episodes}, best_cost={best_cost}, best_xy={best_xy}, avg_reward={avg_reward:.2f}")
    return {
        "feasible": feasible,
        "infeasible": infeasible,
        "best_cost": best_cost,
        "best_xy": best_xy,
        "avg_reward": avg_reward,
    }


def visualize_exploration(model, episodes: int = 5, deterministic: bool = True, save_dir: str = "sb3_logs/trajectories"):
    """Run the model for a few episodes and save x/y trajectories and cost curves.

    Outputs PNG files (requires matplotlib) under save_dir. If matplotlib is not
    available, saves numpy arrays (.npy) so you can plot offline.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    env = ResourceAllocEnv(step_size=10, max_steps=200)

    try:
        import matplotlib.pyplot as plt
        have_plt = True
    except Exception:
        have_plt = False

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        xs, ys, costs = [], [], []
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            xs.append(int(env.x))
            ys.append(int(env.y))
            costs.append(float(env._cost(env.x, env.y)))
            done = terminated or truncated
            steps += 1

        base = os.path.join(save_dir, f"episode_{ep+1}")
        if have_plt:
            try:
                plt.figure(figsize=(6, 4))
                plt.plot(xs, label='x')
                plt.plot(ys, label='y')
                plt.xlabel('step')
                plt.ylabel('units')
                plt.title(f'Episode {ep+1} trajectory (steps={steps})')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(base + '_xy.png')
                plt.close()

                plt.figure(figsize=(6, 3))
                plt.plot(costs, label='cost')
                plt.xlabel('step')
                plt.ylabel('cost')
                plt.title(f'Episode {ep+1} cost (final={costs[-1]:.0f})')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(base + '_cost.png')
                plt.close()
            except Exception:
                have_plt = False

        # always save raw arrays
        np.save(base + '_x.npy', np.array(xs, dtype=np.int32))
        np.save(base + '_y.npy', np.array(ys, dtype=np.int32))
        np.save(base + '_cost.npy', np.array(costs, dtype=np.float32))

        print(f"Saved episode {ep+1} (steps={steps}) to {save_dir}")

    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resource Allocation RL: modes')
    parser.add_argument('--mode', choices=['manual', 'random', 'train', 'viz'], default='train')
    parser.add_argument('--timesteps', type=int, default=200_000, help='total timesteps for training')
    parser.add_argument('--episodes', type=int, default=3, help='episodes for visualization')
    parser.add_argument('--step-size', type=int, default=10, help='env step size in units')
    parser.add_argument('--max-steps', type=int, default=200, help='max steps per episode')
    args = parser.parse_args()

    if args.mode == 'manual':
        manual_mode(step_size=args.step_size, max_steps=args.max_steps)
    elif args.mode == 'random':
        env = ResourceAllocEnv(step_size=args.step_size, max_steps=args.max_steps)
        obs, info = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            env.render()
        env.close()
    elif args.mode == 'train':
        model, best_sol, answer = train_and_eval(total_timesteps=args.timesteps)
        print(f"Trained model. Best solution found: {best_sol}, answer: {answer}")
        # optional quick viz
        # visualize_exploration(model, episodes=args.episodes) for debugging
    elif args.mode == 'viz':
        # For simplicity, we train a short model then visualize; adapt to load a model if needed
        model = train_and_eval(total_timesteps=args.timesteps)
        visualize_exploration(model, episodes=args.episodes)

# Suggested improvements you might want to iterate on:
# - Add a direct-assignment action that sets x or y directly (e.g. action vector for (x,y)).
# - Change step_size to 1 for finer control (longer episodes), or larger for faster search.
# - Implement an action space that proposes (dx,dy) pairs so agent moves faster in 2D.
# - Reward shaping that guides agent toward feasibility first (e.g., reward for satisfying x-y>=200)
# - Provide a continuous action version that outputs real-valued allocations then round to ints.



