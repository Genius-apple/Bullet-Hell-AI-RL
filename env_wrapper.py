import numpy as np
import gym
from gym import spaces
from game import BulletHellGame

class BulletHellEnv(gym.Env):
    """
    OpenAI Gym 兼容的环境包装器
    """
    def __init__(self, render_mode=False):
        super(BulletHellEnv, self).__init__()
        
        self.game = BulletHellGame(render_mode)
        
        # 动作空间: 9 个离散动作
        self.action_space = spaces.Discrete(9)
        
        # 观察空间: 玩家坐标 (2) + 10个最近子弹相对坐标和速度 (40) = 42
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(42,), dtype=np.float32
        )

    def reset(self):
        return self.game.reset()

    def step(self, action):
        state, reward, done, info = self.game.step(action)
        return state, reward, done, info

    def render(self, mode='human'):
        self.game.render()

    def set_info(self, info):
        """传递训练信息到游戏环境"""
        self.game.set_info(info)

    def close(self):
        self.game.close()
