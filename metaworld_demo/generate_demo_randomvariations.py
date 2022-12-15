import metaworld
import random
import  metaworld.policies as policies
import cv2
import numpy as np
from mujoco_py.modder import MaterialModder, TextureModder

import pickle
from pathlib import Path
from collections import deque

from video import VideoRecorder

env_names = ["door-unlock-v2"]
num_demos = 1
exp = 'multicolor'

POLICY = {
	'button-press-v2': policies.SawyerButtonPressV2Policy,
	'button-press-topdown-v2': policies.SawyerButtonPressTopdownV2Policy,
	'drawer-close-v2': policies.SawyerDrawerCloseV2Policy,
	'hammer-v2': policies.SawyerHammerV2Policy,
	'door-open-v2': policies.SawyerDoorOpenV2Policy,
	'drawer-open-v2': policies.SawyerDrawerOpenV2Policy,
	'bin-picking-v2': policies.SawyerBinPickingV2Policy,
	'faucet-open-v2': policies.SawyerFaucetOpenV2Policy,
	'faucet-close-v2': policies.SawyerFaucetCloseV2Policy,
	'door-unlock-v2': policies.SawyerDoorUnlockV2Policy
}

CAMERA = {
	'button-press-v2': 'corner',
	'button-press-topdown-v2': 'corner',
	'drawer-close-v2': 'corner',
	'hammer-v2': 'corner2', #'corner3',
	'door-open-v2': 'corner3',
	'drawer-open-v2': 'corner',
	'bin-picking-v2': 'corner',
	'faucet-open-v2': 'corner',
	'faucet-close-v2': 'corner',
	'door-unlock-v2': 'corner'
}

NUM_STEPS = {
	'button-press-v2': 125,
	'button-press-topdown-v2': 125,
	'drawer-close-v2': 125,
	'hammer-v2': 125,		
	'door-open-v2': 125,
	'drawer-open-v2': 125,
	'bin-picking-v2': 175,
	'faucet-open-v2': 125,
	'faucet-close-v2': 125,
	'door-unlock-v2': 125
}

# print(metaworld.ML1.ENV_NAMES)  # Check out the available environments

for env_name in env_names:
	print(f"################### {env_name} #######################")
	# Initialize policy
	policy = POLICY[env_name]()

	# Initialize env
	ml1 = metaworld.ML1(env_name) # Construct the benchmark, sampling tasks
	env = ml1.train_classes[env_name]()  # Create an environment with task `pick_place`

	# Initialize save dir
	save_dir = Path("./demos") / env_name
	save_dir.mkdir(parents=True, exist_ok=True)

	# Initialize video recorder
	video_recorder = VideoRecorder(save_dir, camera_name=CAMERA[env_name])

	images_list = list()
	observations_list = list()
	actions_list = list()
	rewards_list = list()

	# import ipdb; ipdb.set_trace()
	
	# Get material id to modify color
	# names = ["HammerHandle"] # Hammer
	# names = ["objGeom", "boxGeom"] # DrawerClose
	# names = ["leftbin", "rightbin", "object"] # BinPicking
	# names = ['btnGeom', 'Box'] # ButtonPressTopdown
	# names = ['handle', 'safe'] # DoorOpen
	names = ["lock", "safe"] # DoorUnlock
	geom_ids = [env.sim.model.geom_name2id(name) for name in names]
	mat_ids = [env.sim.model.geom_matid[geom_id] for geom_id in geom_ids]

	# # Make hammer heavier
	# mass_name = 'hammer' # Hammer; change to 10.0
	# mass_name = 'objA' # BinPicking; change to 0.5
	# idx = env.sim.model.body_name2id(mass_name)
	# env.sim.model.body_mass[idx] = 0.5

	episode = 0
	video_recorder.init(env)
	while episode < num_demos:
		print(f"Episode {episode}")
		images = list()
		observations = list()
		actions = list()
		rewards = list()
		image_stack = deque([], maxlen=3)
		goal_achieved = 0

		# Set random goal
		task = ml1.train_tasks[episode] #random.choice(ml1.train_tasks)
		env.set_task(task)  # Set task

		# Set random color
		for mat_id in mat_ids:
			color = np.random.uniform(size=(4,)).astype(np.float32)
			color[3] = 1.0
			env.sim.model.mat_rgba[mat_id] = color
		
		# Set random mass
		# env.sim.model.body_mass[idx] = np.random.randint(1, 5) * 0.1
		# print(f"Mass: {env.sim.model.body_mass[idx]}")

		# Reset env
		observation = env.reset()  # Reset environment
		video_recorder.record(env)
		num_steps = NUM_STEPS[env_name]
		# num_steps = num_steps if num_steps is not None else env.max_path_length
		for step in range(num_steps):
			# Get observation
			observations.append(observation)
			# Get frames
			frame = env.render(offscreen=True, camera_name=CAMERA[env_name])
			frame = cv2.resize(frame, (84,84))
			frame = np.transpose(frame, (2,0,1))
			image_stack.append(frame)
			while(len(image_stack)<3):
				image_stack.append(frame)
			images.append(np.concatenate(image_stack, axis=0))
			# Get action
			action = policy.get_action(observation)
			action = np.clip(action, -1.0, 1.0)
			actions.append(action)
			# Act in the environment
			observation, reward, done, info = env.step(action)
			rewards.append(reward)
			video_recorder.record(env)
			goal_achieved += info['success'] 

		# Store only if trajectory is successful
		episode = episode + 1
		images_list.append(np.array(images))
		observations_list.append(np.array(observations))
		actions_list.append(np.array(actions))
		rewards_list.append(np.array(rewards))
		

	video_recorder.save(f'demo_{exp}.mp4')

	file_path = save_dir / f'expert_demos_{exp}.pkl'
	payload = [images_list, observations_list, actions_list, rewards_list]


	with open(str(file_path), 'wb') as f:
		pickle.dump(payload, f)