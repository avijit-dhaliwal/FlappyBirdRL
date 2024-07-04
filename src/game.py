import pygame
import random
import numpy as np

class FlappyBird:
    def __init__(self, width=400, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Flappy Bird RL")

        self.bird_img = pygame.image.load("assets/bird.jpeg").convert_alpha()
        self.bird_img = pygame.transform.scale(self.bird_img, (40, 30))
        self.pipe_img = pygame.image.load("assets/pipe.png").convert_alpha()
        self.bg_img = pygame.image.load("assets/background.png").convert()
        self.bg_img = pygame.transform.scale(self.bg_img, (width, height))

        self.bird_y = height // 2
        self.bird_velocity = 0
        self.gravity = 0.5
        self.jump_strength = -10

        self.pipe_width = 70
        self.pipe_gap = 150
        self.pipe_x = width
        self.pipe_height = random.randint(100, height - 200)

        self.score = 0
        self.font = pygame.font.Font(None, 36)

    def reset(self):
        self.bird_y = self.height // 2
        self.bird_velocity = 0
        self.pipe_x = self.width
        self.pipe_height = random.randint(100, self.height - 200)
        self.score = 0
        return self.get_state()

    def get_state(self):
        return np.array([
            self.bird_y / self.height,
            self.bird_velocity / 10,
            self.pipe_x / self.width,
            self.pipe_height / self.height,
        ])

    def step(self, action):
        reward = 0.1
        done = False

        if action == 1:
            self.bird_velocity = self.jump_strength

        self.bird_velocity += self.gravity
        self.bird_y += self.bird_velocity
        self.pipe_x -= 5

        if self.pipe_x < -self.pipe_width:
            self.pipe_x = self.width
            self.pipe_height = random.randint(100, self.height - 200)
            self.score += 1
            reward = 1

        if (self.bird_y < 0 or self.bird_y > self.height or
            (self.pipe_x < 50 and (self.bird_y < self.pipe_height or 
             self.bird_y > self.pipe_height + self.pipe_gap))):
            done = True
            reward = -1

        return self.get_state(), reward, done

    def render(self):
        self.screen.blit(self.bg_img, (0, 0))
        self.screen.blit(self.bird_img, (10, int(self.bird_y)))
        self.screen.blit(self.pipe_img, (int(self.pipe_x), 0), 
                         (0, 0, self.pipe_width, self.pipe_height))
        self.screen.blit(self.pipe_img, (int(self.pipe_x), self.pipe_height + self.pipe_gap), 
                         (0, 0, self.pipe_width, self.height - self.pipe_height - self.pipe_gap))

        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()