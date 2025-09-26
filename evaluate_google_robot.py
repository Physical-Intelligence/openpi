import third_party.SimplerEnv.simpler_env as simpler_env
from third_party.SimplerEnv.simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import numpy as np
import imageio
import collections

import openpi.training.config as config
from openpi.policies import policy_config
from openpi.shared import download
from openpi_client import websocket_client_policy as _websocket_client_policy

from tqdm import trange, tqdm

experiment_cfg = {
    "env_name": "google_robot_pick_coke_can",

    "max_timesteps": 50,
    "ntrials": 5,
    "env_seed": 42
}

pi_cfg = {
    "replan_steps": 5,
    "host": "0.0.0.0",
    "port": 8000,
}

def main(experiment_cfg,
         pi_cfg):
    # load simpler env
    env = simpler_env.make(experiment_cfg["env_name"])
    prompt = env.get_language_instruction()
    
    # load openpi client
    client = _websocket_client_policy.WebsocketClientPolicy(pi_cfg["host"],
                                                            pi_cfg["port"])
    
    for i in trange(experiment_cfg["ntrials"]):
        t = 0

        image_seq = []
        action_plan = collections.deque()
        obs, reset_info = env.reset(seed=(experiment_cfg["env_seed"] + i))
        
        while t < experiment_cfg["max_timesteps"]:               
            tcp_pose = np.asarray(obs["extra"]["tcp_pose"], dtype=np.float32)  
            qpos = np.asarray(obs["agent"]["qpos"], dtype=np.float32)
            finger_l, finger_r = qpos[7], qpos[8]
            gripper = 0.5 * (finger_l + finger_r)  
            input_obs = np.concatenate([tcp_pose, [gripper]])

            image = get_image_from_maniskill2_obs_dict(env, obs)

            if not action_plan:
                element = {
                    "state": input_obs,
                    "image": image,
                    "prompt": str(prompt)
                }

                action_chunk = client.infer(element)["actions"]
                action_plan.extend(action_chunk[: pi_cfg["replan_steps"]])
            
            action = action_plan.popleft()
            action = np.array(action, dtype=np.float32, copy=True)

            print("ACTION")
            print(action)
            print("STATE")
            print(input_obs)

            obs, reward, done, truncated, info = env.step(action)

            image_seq.append(image)
            t += 1
        
        imageio.mimwrite(
            f"trial_{i}.mp4",
            [np.asarray(x) for x in image_seq],
            fps=10,
        )
        
if __name__ == "__main__":
    main(experiment_cfg, pi_cfg)