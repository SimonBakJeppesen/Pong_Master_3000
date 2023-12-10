import gym
from gym import spaces
import numpy as np
import torch
import scipy.constants as consts
from gym.envs.registration import register

class RobotArm(gym.Env):
    #metadata = {}
    def __init__(self, upper_arm_length, lower_arm_length, max_acceleration, 
                 free_angles=4, dt=0.05, optimal_time=40, render_mode=None):
        self._free_angles = free_angles
        self._dt = dt
        self._optimal_time = optimal_time
        self._time_step = 0
        self._angles = np.empty((self._free_angles), dtype=np.float32)
        self._velocities = np.empty((self._free_angles), dtype=np.float32)
        self._cup_position = np.empty((3), dtype=np.float32)
        self._upper_arm_length = upper_arm_length
        self._lower_arm_length = lower_arm_length
        self._max_acceleration = max_acceleration

        self.observation_space = spaces.Dict({
            'angles': spaces.Box(low=-np.inf, high=np.inf, shape=(self._free_angles,), dtype=np.float32),
            'velocities': spaces.Box(low=-np.inf, high=np.inf, shape=(self._free_angles,), dtype=np.float32),
            'cup_position': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        })

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,self._free_angles+1), dtype=np.float32)

        #Not implementing rendering for now
        assert render_mode is None
        self.render_mode = render_mode

    def _get_obs(self):
        return {'angles': self._angles, 'velocities': self._velocities, 'cup_position': self._cup_position}

    def _get_info(self):
        return {'time_step': self._time_step, 'secret_info': 'Simon is cute'}

    def reset(self, seed=None):
        super().reset(seed=seed)
        self._time_step = 0
        #self._angles = self.np_random.uniform(low=-np.pi, high=np.pi, size=self._free_angles).astype(np.float32)
        self._angles = np.array([np.pi/2, np.pi/2, np.pi/2, -np.pi/2], dtype=np.float32)
        self._velocities = np.zeros(self._free_angles, dtype=np.float32)
        #self._cup_position = np.append(self.np_random.uniform(low=60, high=80, size=2), -60).astype(np.float32)
        self._cup_position = np.array([70,70,-60], dtype=np.float32)

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        if action[4] > 0:
            terminated = True
            reward = self._get_reward()
        else:
            terminated = False
            reward = 0
        self._time_step += 1
        acceleration = action[:4]*self._max_acceleration
        self._angles += self._velocities * self._dt + 0.5 * acceleration * self._dt**2
        self._velocities += acceleration * self._dt

        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, False, info
    
    def _get_reward(self):
        cup_radius = 3
        ball_landing, _ = self._get_ball_landing()
        cup_distance =  np.linalg.norm(self._cup_position[:2])
        r_b = ball_landing / np.linalg.norm(ball_landing)
        r_k = self._cup_position[:2] / cup_distance

        projection = np.dot(r_b, r_k)
        normed_ball_cup_distance = np.linalg.norm(self._cup_position[:2] - ball_landing) / cup_distance
        exp_normed_distance = np.exp(-normed_ball_cup_distance)
        reward = exp_normed_distance*projection
        return reward
    
    def _get_ball_landing(self):
        position_XYZ, velocities_XYZ = self._get_full_hand_XYZ()
        a = -consts.g*10; b = velocities_XYZ[2]; c = position_XYZ[2]-self._cup_position[2] #g in cm/s^2
        flight_time1 = (-b+np.sqrt(b**2-4*a*c))/(2*a)
        flight_time2 = (-b-np.sqrt(b**2-4*a*c))/(2*a)
        flight_time = max(flight_time1, flight_time2)
        return position_XYZ[:2] + velocities_XYZ[:2]*flight_time, flight_time
    
    def _get_full_hand_XYZ(self):
        'Returns hand position and velocity'
        angles = torch.from_numpy(self._angles.copy())
        angles.requires_grad = True

        _, _, hand_position = self._get_positions_XYZ(angles, [0, self._upper_arm_length, self._lower_arm_length])
        dXYZ_dt = torch.empty(3)
        for XYZ_index in range(3):
            hand_position[XYZ_index].backward(retain_graph=True)
            dXYZ_dt[XYZ_index] = torch.sum(torch.from_numpy(self._velocities.copy())*angles.grad)
            angles.grad.zero_()

        return hand_position.detach().numpy(), dXYZ_dt.numpy()
    
    def _get_positions_XYZ(self, state, d_values):
        c_34, c_44 = d_values[2], 1
        
        B_alpha = state[3] 
        B_theta = state[2]
        B_sin_alpha = torch.sin(B_alpha).unsqueeze(0)
        B_sin_theta = torch.sin(B_theta).unsqueeze(0)
        B_cos_theta = torch.cos(B_theta).unsqueeze(0)
        
        B_col_2 = torch.cat((B_sin_theta*B_sin_alpha, -B_cos_theta*B_sin_alpha, torch.cos(B_alpha).unsqueeze(0), torch.tensor([0.0])), dim=0)
        B_col_3 = torch.tensor([0.0, 0.0, d_values[1], 1.0])
        
        A_alpha = state[1] 
        A_theta = state[0]
        A_sin_alpha = torch.sin(A_alpha).unsqueeze(0)
        A_sin_theta = torch.sin(A_theta).unsqueeze(0)
        A_cos_theta = torch.cos(A_theta).unsqueeze(0)
        A_cos_alpha = torch.cos(A_alpha).unsqueeze(0)
        
        A_row_0 = torch.cat((A_cos_theta, -A_sin_theta*A_cos_alpha, A_sin_theta*A_sin_alpha, torch.tensor([0.0])), dim=0)
        A_row_1 = torch.cat((A_sin_theta, A_cos_theta*A_cos_alpha, -A_cos_theta*A_sin_alpha, torch.tensor([0.0])), dim=0)
        A_row_2 = torch.tensor([0, A_sin_alpha, A_cos_alpha, d_values[0]])
        #In A matrix
        x_1, y_1, z_1 = torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0])
        A = torch.cat((x_1, y_1, z_1), dim=0)
        
        #In A*B matrix
        x_2, y_2, z_2 = torch.dot(A_row_0, B_col_3).unsqueeze(0), torch.dot(A_row_1, B_col_3).unsqueeze(0), torch.dot(A_row_2, B_col_3).unsqueeze(0)
        AB = torch.cat((x_2, y_2, z_2), dim=0)
        
        # In (A*B)*C matrix
        x_3 = c_34 * torch.dot(A_row_0,B_col_2).unsqueeze(0) + c_44 * x_2
        y_3 = c_34 * torch.dot(A_row_1,B_col_2).unsqueeze(0) + c_44 * y_2
        z_3 = c_34 * torch.dot(A_row_2,B_col_2).unsqueeze(0) + c_44 * z_2
        ABC = torch.cat((x_3, y_3, z_3), dim=0)        
        
        return torch.stack((A,AB,ABC), dim=0)
    
register(
    id='RobotArm-v0',
    entry_point='robot_arm_gym:RobotArm',
    max_episode_steps = 200
)