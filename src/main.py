import pygame
from game import FlappyBird
from agent import DQNAgent
from visualizer import ActivationVisualizer

def train_flappy_bird():
    env = FlappyBird()
    state_size = 4
    action_size = 2
    agent = DQNAgent(state_size, action_size)
    visualizer = ActivationVisualizer(agent.model)

    batch_size = 32
    episodes = 1000
    update_target_every = 5

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            env.render()
            visualizer.update_plot(state)

            if done:
                print(f"Episode: {episode + 1}, Score: {env.score}, Total Reward: {total_reward}")
                break

            agent.replay(batch_size)

        if (episode + 1) % update_target_every == 0:
            agent.update_target_model()

    pygame.quit()

if __name__ == "__main__":
    train_flappy_bird()