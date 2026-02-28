# Bullet Hell AI (弹幕躲避游戏 AI)

这是一个基于强化学习 (Reinforcement Learning) 的弹幕躲避游戏 AI 项目。
AI 使用 DQN (Deep Q-Network) 算法学习如何在密集的弹幕中生存，并尽可能获得更高的分数。

游戏环境完全基于 Pygame 开发，不依赖任何第三方闭源游戏，可以在本地 CPU 上流畅运行。

## 🎮 演示

(在这里你可以放一张 AI 玩游戏的 GIF 动图)

## ✨ 特性

- **原创游戏环境**：使用 Pygame 编写的弹幕躲避游戏，无需安装其他复杂环境。
- **强化学习内核**：使用 PyTorch 实现 DQN 算法，包含经验回放 (Replay Buffer) 和目标网络 (Target Network)。
- **实时可视化**：训练过程中可以实时看到 AI 的操作和游戏画面，以及当前的训练数据（Episode, Score, Epsilon）。
- **断点续训**：支持模型保存和加载，训练中断后可以从上次的进度继续训练。
- **擦弹奖励 (Graze System)**：为了鼓励 AI 做出更极限的操作，游戏中加入了“擦弹”机制，近距离躲避子弹会获得更高分数。

## 🛠️ 安装

1.  克隆或下载本项目到本地。
2.  安装依赖库：

```bash
pip install -r requirements.txt
```

主要依赖：
- `torch` (PyTorch)
- `pygame` (游戏引擎)
- `numpy` (数值计算)
- `gym` (强化学习环境接口)
- `matplotlib` (绘图)

## 🚀 快速开始

### 1. 训练 AI

运行 `train.py` 开始训练：

```bash
python train.py
```

- 训练过程中会实时显示游戏窗口。
- 模型会自动保存为 `checkpoint.pth`。
- 按 `Ctrl+C` 可以随时中断训练并保存进度。
- 再次运行 `python train.py` 会自动加载 `checkpoint.pth` 并继续训练。

### 2. 观看 AI 玩游戏 (推理模式)

如果你想看训练好的 AI 是如何玩游戏的，运行 `play.py`：

```bash
python play.py
```

- 这将加载 `checkpoint.pth` 并以 0 探索率 (完全利用) 运行 AI。

## 📂 项目结构

```
bullet_hell_ai/
├── game.py          # 游戏核心逻辑 (Pygame)
├── agent.py         # DQN 智能体实现 (神经网络 + 算法)
├── env_wrapper.py   # 将游戏封装为 OpenAI Gym 格式的环境
├── train.py         # 训练脚本 (包含训练循环、可视化、保存逻辑)
├── play.py          # 推理脚本 (加载模型并演示)
├── requirements.txt # 项目依赖
├── README.md        # 项目说明文档
└── checkpoint.pth   # 训练好的模型文件 (自动生成)
```

## 🧠 算法细节

- **状态空间 (State)**：42 维向量
    - 玩家坐标 (2)
    - 最近 10 个子弹的相对坐标 (20)
    - 最近 10 个子弹的速度向量 (20)
- **动作空间 (Action)**：9 个离散动作 (上下左右、左上、左下、右上、右下、不动)
- **奖励函数 (Reward)**：
    - 存活奖励：每帧 +0.1
    - 擦弹奖励：距离子弹极近时获得额外奖励
    - 死亡惩罚：-10

## 🤝 贡献

欢迎提交 Issue 或 Pull Request 来改进这个项目！

## 📄 许可证

MIT License
