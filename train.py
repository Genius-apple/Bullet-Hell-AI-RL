import numpy as np
import time
import torch
import pygame
from collections import deque
import matplotlib.pyplot as plt
from env_wrapper import BulletHellEnv
from agent import DQNAgent

def train():
    # 初始化环境
    # render_mode=False 可以加速训练，True 可以看到游戏画面
    env = BulletHellEnv(render_mode=True)
    
    # 状态和动作维度
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # 初始化智能体
    agent = DQNAgent(state_size, action_size, seed=0)
    
    # 训练参数
    n_episodes = 2000
    max_t = 1000
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 0.995
    
    scores = []                        # 记录每回合分数
    scores_window = deque(maxlen=100)  # 最近100回合的平均分
    eps = eps_start                    # 初始化 epsilon
    start_episode = 1                  # 默认从第 1 回合开始

    # 尝试加载已有模型
    import os
    if os.path.exists('checkpoint.pth'):
        print("发现已有模型 checkpoint.pth，正在加载...")
        try:
            checkpoint = torch.load('checkpoint.pth', map_location='cpu')
            
            # 检查是旧格式（只保存了权重）还是新格式（保存了完整状态）
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                agent.qnetwork_local.load_state_dict(checkpoint['state_dict'])
                agent.qnetwork_target.load_state_dict(checkpoint['state_dict'])
                start_episode = checkpoint['episode'] + 1
                scores = checkpoint.get('scores', [])
                scores_window = deque(checkpoint.get('scores_window', []), maxlen=100)
                eps = checkpoint.get('epsilon', eps)
                print(f"成功恢复训练进度！将从第 {start_episode} 回合继续，当前 Epsilon: {eps:.2f}")
                
                # 如果已经达到或超过预设的最大回合数，自动增加 1000 回合
                if start_episode > n_episodes:
                    print(f"注意：之前的训练已达到 {n_episodes} 回合。自动增加 1000 回合以继续训练。")
                    n_episodes = start_episode + 1000
                    
            else:
                # 兼容旧格式
                agent.qnetwork_local.load_state_dict(checkpoint)
                agent.qnetwork_target.load_state_dict(checkpoint)
                eps = 0.5 # 旧格式没有保存 epsilon，给一个中间值
                print(f"成功加载模型权重（旧格式）！将从第 1 回合开始，Epsilon 重置为 {eps:.2f}")
                
        except Exception as e:
            print(f"加载模型失败: {e}，将从头开始训练。")
    
    print("开始训练 AI Bullet Hell...")
    print("按 Ctrl+C 可以停止训练并保存模型")
    
    try:
        for i_episode in range(start_episode, n_episodes+1):
            state = env.reset()
            score = 0
            for t in range(max_t):
                # 选择动作
                action = agent.act(state, eps)
                
                # 执行动作
                next_state, reward, done, _ = env.step(action)
                
                # 学习
                agent.step(state, action, reward, next_state, done)
                
                state = next_state
                score += reward
                
                # 渲染
                env.set_info({
                    "Episode": i_episode,
                    "Avg Score": f"{np.mean(scores_window):.1f}" if scores_window else 0,
                    "Epsilon": f"{eps:.2f}"
                })
                env.render()
                
                if done:
                    break
            
            # 更新分数记录
            scores_window.append(score)
            scores.append(score)
            
            # 衰减 epsilon
            eps = max(eps_end, eps_decay*eps)
            
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tEpsilon: {eps:.2f}', end="")
            
            if i_episode % 100 == 0:
                print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')
                # 保存模型
                checkpoint = {
                    'state_dict': agent.qnetwork_local.state_dict(),
                    'episode': i_episode,
                    'scores': scores,
                    'scores_window': list(scores_window),
                    'epsilon': eps
                }
                torch.save(checkpoint, 'checkpoint.pth')
                
    except KeyboardInterrupt:
        print("\n训练中断，保存模型...")
        checkpoint = {
            'state_dict': agent.qnetwork_local.state_dict(),
            'episode': i_episode,
            'scores': scores,
            'scores_window': list(scores_window),
            'epsilon': eps
        }
        torch.save(checkpoint, 'checkpoint.pth')
        
    finally:
        env.close()
        print("训练结束")

        # 绘制分数曲线
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()

if __name__ == "__main__":
    train()
