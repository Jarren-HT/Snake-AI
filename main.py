import pygame
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time


pygame.init()

# Game constants

import os

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 1200
BLOCK_SIZE = 50
SNAKE_SPEED = 10

# Colors
BLACK = (0, 0, 0)
GREEN = (57, 218, 138)
RED = (242, 62, 62)
YELLOW = (255, 255, 102)
WHITE = (255, 255, 255)

x = 100
y = 45
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (x,y)

# Snake class
class Snake:
    def __init__(self):
        self.length = 3
        self.positions = [(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)]
        self.direction = "RIGHT"

    def move(self):
        head_x, head_y = self.positions[0]
        if self.direction == "UP":
            new_head = (head_x, head_y - BLOCK_SIZE)
        elif self.direction == "DOWN":
            new_head = (head_x, head_y + BLOCK_SIZE)
        elif self.direction == "LEFT":
            new_head = (head_x - BLOCK_SIZE, head_y)
        elif self.direction == "RIGHT":
            new_head = (head_x + BLOCK_SIZE, head_y)
        self.positions.insert(0, new_head)
        if len(self.positions) > self.length:
            self.positions.pop()

    def change_direction(self, new_direction):
        opposite_directions = {
            "UP": "DOWN", "DOWN": "UP",
            "LEFT": "RIGHT", "RIGHT": "LEFT"
        }
        if new_direction != opposite_directions.get(self.direction):
            self.direction = new_direction

    def draw(self, window):
        for pos in self.positions:
            pygame.draw.rect(window, GREEN, (pos[0], pos[1], BLOCK_SIZE, BLOCK_SIZE))

    def check_collision(self):
        head = self.positions[0]
        return (
            head[0] < 0 or head[0] >= WINDOW_WIDTH or
            head[1] < 0 or head[1] >= WINDOW_HEIGHT or
            head in self.positions[1:]
        )

# Apple class
class Apple:
    def __init__(self):
        self.position = self.generate_random_position()

    def generate_random_position(self):
        x = random.randint(0, (WINDOW_WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (WINDOW_HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        return x, y

    def draw(self, window):
        pygame.draw.rect(window, RED, (self.position[0], self.position[1], BLOCK_SIZE, BLOCK_SIZE))

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, num_states, num_actions, learning_rate, discount_factor, epsilon):
        self.q_table = np.zeros((num_states, num_actions))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.memory = deque(maxlen=1000)  # Replay buffer

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            current_q = self.q_table[state, action]
            max_future_q = np.max(self.q_table[next_state]) if not done else 0
            self.q_table[state, action] += self.learning_rate * (
                reward + self.discount_factor * max_future_q - current_q
            )

    def get_action(self, state):
        if np.random.rand() < self.epsilon:  # Exploration
            return np.random.randint(4)
        return np.argmax(self.q_table[state])  # Exploitation

    def train(self, state, action, reward, next_state, done):
        current_q = self.q_table[state, action]
        max_future_q = np.max(self.q_table[next_state]) if not done else 0
        self.q_table[state, action] += self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)

# State representation function
def get_state(snake, apple):
    head_x, head_y = snake.positions[0]
    apple_x, apple_y = apple.position
    danger_up = (head_y - BLOCK_SIZE, head_x) in snake.positions or head_y == 0
    danger_down = (head_y + BLOCK_SIZE, head_x) in snake.positions or head_y == WINDOW_HEIGHT - BLOCK_SIZE
    danger_left = (head_x - BLOCK_SIZE, head_y) in snake.positions or head_x == 0
    danger_right = (head_x + BLOCK_SIZE, head_y) in snake.positions or head_x == WINDOW_WIDTH - BLOCK_SIZE

    return (
        (head_x > apple_x) * 1 +
        (head_x < apple_x) * 2 +
        (head_y > apple_y) * 4 +
        (head_y < apple_y) * 8 +
        danger_up * 16 +
        danger_down * 32 +
        danger_left * 64 +
        danger_right * 128
    )

def update_plot(ax, score_history, rolling_avg_scores):
    ax.clear()
    ax.set_title("Training Performance")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Score")
    ax.plot(range(len(score_history)), score_history, label="Score")
    ax.plot(range(len(rolling_avg_scores)), rolling_avg_scores, label="Avg Score (Rolling)")
    ax.set_xlim(0, len(score_history))  # Set x-axis dynamically
    ax.legend()


# Game loop
def game_loop():
    plt.ion()
    fig, ax = plt.subplots()
    score_history = deque(maxlen=100)
    rolling_avg_scores = deque(maxlen=100)

    clock = pygame.time.Clock()
    snake = Snake()
    apple = Apple()

    num_states = 256
    num_actions = 4
    agent = QLearningAgent(
        num_states=num_states,
        num_actions=num_actions,
        learning_rate=0.0989,
        discount_factor=0.97598,
        epsilon=0.9
    )
    agent.epsilon_decay = 0.99

    iteration = 0
    global SNAKE_SPEED

    frame_times = deque(maxlen=100)  # Store recent frame times

    while True:
        snake = Snake()
        apple = Apple()
        score = 0
        game_over = False

        while not game_over:
            start_time = time.time()  # Start timing this frame

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    plt.ioff()
                    plt.savefig("training_performance.png")
                    with open("frame_time_log.txt", "w") as f:
                        for t in frame_times:
                            f.write(f"{t:.4f}\n")
                    return
                elif event.type == pygame.KEYDOWN:
                    global SNAKE_SPEED
                    if event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        SNAKE_SPEED = min(SNAKE_SPEED + 50, 15000)
                    elif event.key == pygame.K_MINUS:
                        SNAKE_SPEED = max(SNAKE_SPEED - 50, 1)

            state = get_state(snake, apple)
            action = agent.get_action(state)
            if action == 0: snake.change_direction("UP")
            elif action == 1: snake.change_direction("DOWN")
            elif action == 2: snake.change_direction("LEFT")
            elif action == 3: snake.change_direction("RIGHT")

            snake.move()
            reward = -0.1

            head_x, head_y = snake.positions[0]
            apple_x, apple_y = apple.position

            if snake.positions[0] == apple.position:
                snake.length += 1
                apple.position = apple.generate_random_position()
                reward = 10
                score += 50
            else:
                new_head = snake.positions[0]
                current_distance = abs(new_head[0] - apple_x) + abs(new_head[1] - apple_y)
                previous_distance = abs(head_x - apple_x) + abs(head_y - apple_y)
                if current_distance < previous_distance:
                    reward += 1.5

            if snake.check_collision():
                reward = -100
                game_over = True

            next_state = get_state(snake, apple)
            agent.train(state, action, reward, next_state, game_over)

            window.fill(BLACK)
            snake.draw(window)
            apple.draw(window)

            font = pygame.font.Font(None, 24)
            iteration_text = font.render(f"Iteration: {iteration}", True, WHITE)
            cur_score_text = font.render(f"Score: {score}", True, WHITE)
            average_score_text = font.render(f"Avg Score: {np.mean(score_history) if score_history else 0:.2f}", True, WHITE)
            speed_text = font.render(f"Speed: {SNAKE_SPEED}", True, WHITE)

            window.blit(cur_score_text, (10, 90))
            window.blit(speed_text, (10, 70))
            window.blit(iteration_text, (10, 10))
            window.blit(average_score_text, (10, 30))

            decision_overlay = ["up", "down", "left", "right"][action]
            decision_text = font.render(f"Decision: {decision_overlay}", True, WHITE)
            window.blit(decision_text, (10, 50))

            pygame.display.flip()

            clock.tick(SNAKE_SPEED)

            # End timing and calculate frame time
            end_time = time.time()
            frame_time = (end_time - start_time) * 1000  # Convert to milliseconds
            frame_times.append(frame_time)

        iteration += 1
        score_history.append(score)
        rolling_avg_scores.append(np.mean(score_history))
        # update_plot(ax, score_history, rolling_avg_scores)
        agent.epsilon = max(0.1, agent.epsilon * agent.epsilon_decay)

        # Display average frame time
        if len(frame_times) > 0 and iteration % 10 == 0:  # Update every 10 iterations
            avg_frame_time = np.mean(frame_times)
            print(f"Average Frame Time: {avg_frame_time:.2f} ms, FPS: {1000 / avg_frame_time:.2f}")
            print(agent.epsilon)

# Run the game
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Snake Game with Q-Learning")
game_loop()
plt.savefig('Personal/SnakeAi/DataPictures/data.png')
