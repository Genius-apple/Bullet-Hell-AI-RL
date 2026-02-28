import pygame
import random
import math
import numpy as np

# 游戏参数
WIDTH, HEIGHT = 600, 600
PLAYER_SIZE = 20
BULLET_SIZE = 10
FPS = 60

# 颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

class BulletHellGame:
    def __init__(self, render_mode=True):
        self.render_mode = render_mode
        self.info = {}  # 存储额外显示信息
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("AI Bullet Hell")
            self.clock = pygame.time.Clock()
        
        self.reset()

    def reset(self):
        # 玩家初始位置
        self.player_pos = [WIDTH // 2, HEIGHT // 2]
        self.player_vel = 5
        
        # 子弹列表 [x, y, dx, dy]
        self.bullets = []
        self.score = 0
        self.frame_count = 0
        self.game_over = False
        
        # 奖励机制相关的变量
        self.prev_dist_to_nearest = 0
        
        # 状态空间: 玩家位置 + 最近N个子弹的位置和速度
        return self.get_state()

    def step(self, action):
        """
        Action space:
        0: 停止
        1: 上
        2: 下
        3: 左
        4: 右
        5: 左上
        6: 右上
        7: 左下
        8: 右下
        """
        if self.game_over:
            return self.get_state(), 0, True, {}

        # 1. 玩家移动
        dx, dy = 0, 0
        if action == 1: dy = -self.player_vel
        elif action == 2: dy = self.player_vel
        elif action == 3: dx = -self.player_vel
        elif action == 4: dx = self.player_vel
        elif action == 5: dx, dy = -self.player_vel, -self.player_vel
        elif action == 6: dx, dy = self.player_vel, -self.player_vel
        elif action == 7: dx, dy = -self.player_vel, self.player_vel
        elif action == 8: dx, dy = self.player_vel, self.player_vel
        
        # 边界检查
        self.player_pos[0] = max(0, min(WIDTH - PLAYER_SIZE, self.player_pos[0] + dx))
        self.player_pos[1] = max(0, min(HEIGHT - PLAYER_SIZE, self.player_pos[1] + dy))

        # 2. 生成子弹 (每 10 帧生成一个)
        if self.frame_count % 10 == 0:
            self.spawn_bullet()
            
        # 随着时间推移，难度增加 (生成频率加快)
        if self.frame_count > 1000 and self.frame_count % 30 == 0:
             self.spawn_bullet() # 额外生成

        # 3. 更新子弹位置
        active_bullets = []
        for b in self.bullets:
            b[0] += b[2]
            b[1] += b[3]
            # 移除出界的子弹
            if 0 <= b[0] <= WIDTH and 0 <= b[1] <= HEIGHT:
                active_bullets.append(b)
        self.bullets = active_bullets

        # 4. 碰撞检测
        reward = 1.0 # 存活奖励
        hit = False
        
        # 计算最近子弹距离 (用于擦弹奖励)
        min_dist = float('inf')
        
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], PLAYER_SIZE, PLAYER_SIZE)
        
        for b in self.bullets:
            bullet_rect = pygame.Rect(b[0], b[1], BULLET_SIZE, BULLET_SIZE)
            if player_rect.colliderect(bullet_rect):
                hit = True
                break
            
            # 计算中心点距离
            dist = math.sqrt((self.player_pos[0] - b[0])**2 + (self.player_pos[1] - b[1])**2)
            if dist < min_dist:
                min_dist = dist
        
        if hit:
            self.game_over = True
            reward = -100.0
        else:
            # 擦弹奖励: 如果非常接近子弹但没碰到，给予额外奖励
            if min_dist < 40: # 接近距离
                reward += 2.0
            
            # 引导奖励: 鼓励往屏幕中心靠 (避免死角)
            # dist_to_center = math.sqrt((self.player_pos[0] - WIDTH/2)**2 + (self.player_pos[1] - HEIGHT/2)**2)
            # reward -= dist_to_center * 0.001 

        self.score += 1
        self.frame_count += 1
        
        return self.get_state(), reward, self.game_over, {"score": self.score}

    def spawn_bullet(self):
        # 从四边随机生成
        side = random.choice(['top', 'bottom', 'left', 'right'])
        speed = random.uniform(2, 5 + self.frame_count / 500) # 速度随时间增加
        
        if side == 'top':
            x = random.randint(0, WIDTH)
            y = 0
            # 指向玩家的大致方向，加点随机扰动
            angle = math.atan2(self.player_pos[1] - y, self.player_pos[0] - x) + random.uniform(-0.2, 0.2)
        elif side == 'bottom':
            x = random.randint(0, WIDTH)
            y = HEIGHT
            angle = math.atan2(self.player_pos[1] - y, self.player_pos[0] - x) + random.uniform(-0.2, 0.2)
        elif side == 'left':
            x = 0
            y = random.randint(0, HEIGHT)
            angle = math.atan2(self.player_pos[1] - y, self.player_pos[0] - x) + random.uniform(-0.2, 0.2)
        elif side == 'right':
            x = WIDTH
            y = random.randint(0, HEIGHT)
            angle = math.atan2(self.player_pos[1] - y, self.player_pos[0] - x) + random.uniform(-0.2, 0.2)
            
        dx = speed * math.cos(angle)
        dy = speed * math.sin(angle)
        self.bullets.append([x, y, dx, dy])

    def get_state(self):
        # 简单的状态表示: 
        # 1. 玩家归一化坐标 (2)
        # 2. 最近的 10 个子弹的相对坐标和速度 (10 * 4)
        # 3. 如果不足 10 个，补 0
        
        state = []
        # 玩家位置归一化
        state.append(self.player_pos[0] / WIDTH)
        state.append(self.player_pos[1] / HEIGHT)
        
        # 寻找最近的子弹
        # 计算所有子弹到玩家的距离
        dists = []
        for b in self.bullets:
            d = (self.player_pos[0] - b[0])**2 + (self.player_pos[1] - b[1])**2
            dists.append((d, b))
        
        # 排序
        dists.sort(key=lambda x: x[0])
        
        # 取前 10 个
        nearest_bullets = [x[1] for x in dists[:10]]
        
        for b in nearest_bullets:
            # 相对位置归一化
            state.append((b[0] - self.player_pos[0]) / WIDTH)
            state.append((b[1] - self.player_pos[1]) / HEIGHT)
            # 速度归一化 (假设最大速度 10)
            state.append(b[2] / 10.0)
            state.append(b[3] / 10.0)
            
        # 补全
        while len(state) < 2 + 10 * 4:
            state.extend([0, 0, 0, 0])
            
        return np.array(state, dtype=np.float32)

    def render(self):
        if not self.render_mode:
            return
            
        # 检查事件 (防止窗口无响应)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.game_over = True
                pygame.quit()
                return

        self.screen.fill(BLACK)
        
        # 画玩家
        pygame.draw.rect(self.screen, GREEN, (self.player_pos[0], self.player_pos[1], PLAYER_SIZE, PLAYER_SIZE))
        
        # 画子弹
        for b in self.bullets:
            pygame.draw.rect(self.screen, RED, (b[0], b[1], BULLET_SIZE, BULLET_SIZE))
            
        # 显示分数
        font = pygame.font.SysFont(None, 36)
        text = font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(text, (10, 10))
        
        pygame.display.flip()
        self.clock.tick(FPS)

    def set_info(self, info):
        """更新显示信息"""
        self.info.update(info)

    def close(self):
        if self.render_mode:
            pygame.quit()

if __name__ == "__main__":
    # 人类测试模式
    game = BulletHellGame()
    running = True
    while running:
        action = 0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action = 1
        if keys[pygame.K_DOWN]: action = 2
        if keys[pygame.K_LEFT]: action = 3
        if keys[pygame.K_RIGHT]: action = 4
        # 简单的组合键处理 (实际需要更完善)
        if keys[pygame.K_LEFT] and keys[pygame.K_UP]: action = 5
        
        state, reward, done, _ = game.step(action)
        game.render()
        if done:
            print(f"Game Over! Score: {game.score}")
            game.reset()
