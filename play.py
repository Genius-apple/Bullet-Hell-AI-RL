import time
import torch
import numpy as np
from env_wrapper import BulletHellEnv
from agent import DQNAgent

def play():
    # 初始化环境
    env = BulletHellEnv(render_mode=True)
    
    # 状态和动作维度
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # 初始化智能体
    agent = DQNAgent(state_size, action_size, seed=0)
    
    # 加载模型
    model_path = 'checkpoint.pth'
    try:
        # map_location='cpu' 确保在 CPU 上也能加载（即使是在 GPU 上训练的）
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 兼容新旧格式
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
             agent.qnetwork_local.load_state_dict(checkpoint['state_dict'])
        else:
             agent.qnetwork_local.load_state_dict(checkpoint)
             
        print(f"成功加载模型 {model_path}")
    except FileNotFoundError:
        print(f"未找到模型文件 {model_path}，请先运行 train.py 进行训练")
        return
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    print("开始运行演示模式... (按 Ctrl+C 退出)")
    
    try:
        while True:
            state = env.reset()
            score = 0
            done = False
            
            while not done:
                # 选择动作 (epsilon=0 表示完全贪婪，只利用不探索)
                action = agent.act(state, eps=0.0)
                
                # 执行动作
                next_state, reward, done, _ = env.step(action)
                
                state = next_state
                score += reward
                
                # 渲染
                env.set_info({
                    "Mode": "AI Play",
                    "Score": f"{score:.1f}"
                })
                env.render()
                
                # 稍微控制一下速度，如果是回放可能不需要 sleep，取决于电脑性能
                # time.sleep(0.01) 
                
            print(f"Game Over. Score: {score:.1f}")
            time.sleep(1) # 死亡后暂停一秒
            
    except KeyboardInterrupt:
        print("演示结束")
    finally:
        env.close()

if __name__ == "__main__":
    play()
