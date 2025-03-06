import sapien
from mani_skill import ASSET_DIR
from mani_skill.utils import sapien_utils, common
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.building import actors
from bimanual_allegro import Bimanual_Allegro
from mani_skill.utils.building.ground import build_ground
from mani_skill.envs.utils.randomization.pose import random_quaternions
from sapien.core import pysapien
from typing import Union, Any, Dict
import torch
import numpy as np

from mani_skill.utils.io_utils import load_json
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils

from mani_skill.utils.structs import Actor

@register_env("Bimanual_Allegro_YCB", max_episode_steps=200)
class Env(BaseEnv):
    SUPPORTED_ROBOTS = ["Bimanual_Allegro"]

    agent: Union[Bimanual_Allegro]

    def __init__(
        self, *args,
        robot_uids="Bimanual_Allegro",
        robot_init_qpos_noise=0.02,
        **kwargs
    ):
        # # load all YCB model IDs before initializion
        # self.all_model_ids = np.array(
        #     list(
        #         load_json(ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json").keys()
        #     )
        # )

        all_possible_models = list(
            load_json(ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json").keys()
        )

        print(f"Total available YCB objects: {len(all_possible_models)}")

        # select subset of YCB objects
        self.all_model_ids = np.array(all_possible_models[0:1])

        print(f"Using {len(self.all_model_ids)} YCB objects:")
        for i, model_id in enumerate(self.all_model_ids):
            print(f"{i+1}. {model_id}")


        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.table_height = 1.1
        self.initialized = False

        self.right_hand_link = []
        self.left_hand_link = []
        self.left_hand_tip_link = []
        self.right_hand_tip_link = []

        self.max_reward = 5.0

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

        links = self.agent.robot.links
        for link in links:
            if ".0" in link.name and "r" in link.name and "tip" not in link.name:
                self.right_hand_link.append(link)
            elif ".0" in link.name and "r" not in link.name and "tip" not in link.name:
                self.left_hand_link.append(link)
            elif ".0" in link.name and "r" in link.name and "tip" in link.name:
                self.right_hand_tip_link.append(link)
            elif ".0" in link.name and "r" not in link.name and "tip" in link.name:
                self.left_hand_tip_link.append(link)
        self.initialized = True

    def reconfigure(self):
        """Called by the training loop to reconfigure environment"""
        if hasattr(self.scene, 'reconfigure'):
            self.scene.reconfigure()
        # After reconfiguration, re-initialize the episode
        self._initialize_episode(torch.arange(self.num_envs, device=self.device), {})

    def _load_scene(self, options: dict):
        build_ground(self.scene)

        table_builder = self.scene.create_actor_builder()
        table_builder.add_box_visual(
            pose=sapien.Pose(p=[0, 0.5, self.table_height]),
            half_size=[0.7, 0.5, 0.02],
            material=sapien.render.RenderMaterial(
                base_color=[0.1, 0.1, 0.1, 1],
            ),
        )
        table_builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0.5, self.table_height]),
        	half_size=[0.7, 0.5, 0.02],
            material=sapien.physx.PhysxMaterial(1.4, 0.3, 0.1),
        )
        self.table = table_builder.build_static(name="table")

        # sample YCB objects for each parallel environment
        model_ids = self._batched_episode_rng.choice(self.all_model_ids)
        
        stored_actors = []
        for i, model_id in enumerate(model_ids):
            builder = actors.get_actor_builder(self.scene, id=f"ycb:{model_id}")
            builder.set_scene_idxs([i])
            stored_actors.append(builder.build(name=f"{model_id}-{i}"))

        # merge actors for efficient handling
        self.ycb_object = Actor.merge(stored_actors, name="ycb_object")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self._initialize_agent(env_idx)
        self._initialize_actor(env_idx)

    def _initialize_actor(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            b = len(env_idx)
            #print(f"Initializing {b} YCB objects")
            
            # Set initial position to middle of desired range
            # x: default position on table (0.0)
            # y: default position on table (0.5)
            # z: slightly above table surface
            ycb_xyz = torch.tensor([0.0, 0.5, self.table_height + 0.05]).repeat(b, 1)
            #print(f"Initial positions: {ycb_xyz}")
            
            # Generate random offsets for x and y coordinates
            x_rand = -0.2 #np.random.uniform(-0.5, 0.5)
            y_rand = 0.25 # np.random.uniform(-0.2, 0.25)

            # Adjust multipliers to match desired range
            # x: random range of -0.7 to 0.7 based on x_rand
            # y: random range of -0.4 to 0.25 based on y_rand
            # z: no randomness, keep it 0
            ycb_xyz = (torch.rand((b, 3)) * 0.5) * torch.tensor([x_rand, y_rand, 0.0]) + ycb_xyz
        
            # Set z to table height + 0.1 to avoid clipping through the table
            ycb_xyz[:, 2] = self.table_height + 0.1

            # # Generate random orientations for the YCB object
            # orn = random_quaternions(b, device=self.device)

            # Create upright orientation with only rotation around vertical axis
            # This keeps cylinders standing up rather than on their sides
            
            # Start with identity quaternion (w=1, x=y=z=0) - upright orientation
            orn = torch.zeros((b, 4), device=self.device)
            orn[:, 0] = 1.0  # w component = 1 (identity rotation)
            
            # Apply random rotation only around z-axis (vertical axis)
            # This will keep objects upright but with random facing direction
            angle = torch.rand(b, device=self.device) * (2 * np.pi)  # Random angle 0-2π
            
            # Convert angle to quaternion representing rotation around z-axis
            # Format: [w, x, y, z]
            orn[:, 0] = torch.cos(angle / 2)  # w component
            orn[:, 3] = torch.sin(angle / 2)  # z component (rotation around z-axis)

            ycb_pose = Pose.create_from_pq(p=ycb_xyz, q=orn)
            self.ycb_object.set_pose(ycb_pose)

            #print(f"Final YCB poses: {self.ycb_object.pose.p}")

    def _initialize_agent(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            b = len(env_idx)
            
            # create full qpos array (44 total for initialization)
            init_qpos = torch.tensor(
                self.agent.keyframes["init"].qpos, dtype=torch.float32
            ).repeat(b, 1)

            # get indices for right arm/hand joints
            right_indices = []
            for i, joint in enumerate(self.agent.robot.get_active_joints()):
                if "_r" in joint.name:  # only right side joints
                    right_indices.append(i)

            # only add noise to right arm indices
            init_qpos[:, right_indices] += (
                torch.randn((b, len(right_indices)), device=self.device) * 
                self.robot_init_qpos_noise
            )
            
            self.agent.reset(init_qpos)
            self.agent.robot.set_pose(self.agent.keyframes["init"].pose)

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            ycb_pos=self.ycb_object.pose.p,
            ycb_q=self.ycb_object.pose.q,
        )
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                ycb_ppos=self.ycb_object.pose.p,
                ycb_q=self.ycb_object.pose.q,
            )
        return obs

    def evaluate(self):
        if self.initialized:
            right_hand_link_z = torch.concat(
                [
                    link.pose.p[..., 2].unsqueeze(1)
                    for link in self.right_hand_link + self.right_hand_tip_link
                ],
                dim=1,
            )

            fail_collision_table = (right_hand_link_z < self.table_height + 0.02).any(
                dim=1
            )
            fail_ycb_fall = self.ycb_object.pose.p[:, 2] < self.table_height - 0.02
            fail = fail_collision_table | fail_ycb_fall
        else:
            fail = torch.zeros_like(self.ycb_object.pose.p[:, 0], dtype=torch.bool)
            
        # create success condition
        success = self.ycb_object.pose.p[:, 2] >= 1.25
        
        # calculate reward directly here instead of calling compute_dense_reward
        reward = torch.zeros_like(self.ycb_object.pose.p[:, 0], device=self.device)
        reward[success] = self.max_reward
        reward[fail] = -self.max_reward / 4
        
        state = {
            "success": success,
            "fail": fail,
            "episode": {"r": reward}
        }
        return state

    def compute_dense_reward(self, obs: Any, action: np.ndarray, info: Dict):
        batch_size = self.ycb_object.pose.p.shape[0]
        total_reward = torch.zeros(batch_size, device=self.device)
        
        # Get object position
        ycb_xyz = self.ycb_object.pose.p
        
        # Get fingertip positions
        right_hand_tip_positions = torch.cat(
            [link.pose.p.unsqueeze(1) for link in self.right_hand_tip_link],
            dim=1
        )
        hand_center = right_hand_tip_positions.mean(dim=1)
        
        # 1. Approach reward (hand moves close to object)
        distance = torch.linalg.norm(hand_center - ycb_xyz, dim=-1)
        approach_reward = torch.exp(-5.0 * distance)
        total_reward += approach_reward * 1.0
        
        # 2. Grasp formation reward
        fingertip_distances = torch.linalg.norm(
            right_hand_tip_positions - ycb_xyz.unsqueeze(1), dim=-1
        )
        grasp_reward = torch.exp(-10.0 * fingertip_distances.mean(dim=1))
        total_reward += grasp_reward * 2.0
        
        # 3. Lifting reward (progression towards goal of 15cm lift)
        height_above_table = ycb_xyz[:, 2] - self.table_height
        
        # I set 3 levels of lift rewards:
        # Initial lift (0-5cm): basic reward to get robot started
        # Mid lift (5-10cm): increased reward to encourage lifting higher
        # High lift (10-15cm): highest reward as robot is close to goal
        
        # Track if lifting has occurred (for printing)
        if not hasattr(self, "lift_detected"):
            self.lift_detected = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            self.env_steps = torch.zeros(batch_size, device=self.device)
        
        self.env_steps += 1
        newly_lifted = (height_above_table > 0.03) & (~self.lift_detected)
        self.lift_detected = self.lift_detected | (height_above_table > 0.03)
        
        if newly_lifted[0]:
            print(f"\n(ROBOT LIFTING OBJECT) Object lifted to {height_above_table[0].item():.4f}m at step {self.env_steps[0].item():.0f}!")
        
        # Basic lift reward level (linear with height)
        base_lift_reward = torch.clamp(height_above_table * 5.0, min=0.0)
        total_reward += base_lift_reward
        
        # Mid lift reward level (5cm)
        mid_threshold = 0.05
        mid_lift_bonus = torch.where(
            height_above_table > mid_threshold,
            torch.ones_like(height_above_table) * 1.0,  # small bonus reward
            torch.zeros_like(height_above_table)
        )
        total_reward += mid_lift_bonus
        
        # High lift reward level (10cm)
        high_threshold = 0.10
        high_lift_bonus = torch.where(
            height_above_table > high_threshold,
            torch.ones_like(height_above_table) * 2.0,  # mid bonus reward
            torch.zeros_like(height_above_table)
        )
        total_reward += high_lift_bonus
        
        # Success threshold (approaching 15cm)
        goal_threshold = 0.145  # 14.5 cm
        goal_approach_bonus = torch.where(
            height_above_table > goal_threshold,
            torch.ones_like(height_above_table) * 5.0, # large bonus reward
            torch.zeros_like(height_above_table)
        )
        total_reward += goal_approach_bonus
        
        # 4. Penalty for left arm movement (added just in case, might not be needed tbh)
        left_hand_positions = torch.stack([link.pose.p for link in self.left_hand_link], dim=1)
        if not hasattr(self, "initial_left_hand_positions"):
            self.initial_left_hand_positions = left_hand_positions.detach().clone()
        
        left_movement = torch.linalg.norm(
            left_hand_positions - self.initial_left_hand_positions, dim=-1
        ).mean(dim=-1)
        total_reward -= left_movement * 5.0
        
        # Success and failure conditions
        total_reward = torch.clamp(total_reward, -self.max_reward, self.max_reward)
        total_reward[info["success"]] = self.max_reward
        total_reward[info["fail"]] = -self.max_reward / 4
        
        return total_reward

    def compute_normalized_dense_reward(self, obs: Any, action: np.ndarray, info: Dict):
        self.max_reward = 5.0
        dense_reward = self.compute_dense_reward(obs=obs, action=action, info=info)
        norm_dense_reward = dense_reward / (2 * self.max_reward) + 0.5
        return norm_dense_reward
    
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.5, 1.5, 2.0], 
                                    target=[0.0, 0.5, self.table_height])
        print(f"Camera pose: {pose}")
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.5, 1.5, 2.0], 
                                    target=[0.0, 0.5, self.table_height])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )
