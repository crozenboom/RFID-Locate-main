import pygame
import random
import asyncio
import platform

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 400
PADDLE_WIDTH = 15
PADDLE_HEIGHT = 50
BALL_SIZE = 10
BALL_SPEED = 7
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Paddle class
class Paddle:
    def __init__(self):
        self.x = 0
        self.y = (WINDOW_HEIGHT - PADDLE_HEIGHT) // 2
        self.rect = pygame.Rect(self.x, self.y, PADDLE_WIDTH, PADDLE_HEIGHT)

    def move(self, mouse_y):
        # Update paddle position based on mouse Y, constrained to screen
        self.y = max(0, min(WINDOW_HEIGHT - PADDLE_HEIGHT, mouse_y - PADDLE_HEIGHT // 2))
        self.rect.y = self.y

# Ball class
class Ball:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = WINDOW_WIDTH // 2
        self.y = WINDOW_HEIGHT // 2
        self.dx = BALL_SPEED * random.choice((1, -1))
        self.dy = BALL_SPEED * random.choice((1, -1))
        self.rect = pygame.Rect(self.x - BALL_SIZE // 2, self.y - BALL_SIZE // 2, BALL_SIZE, BALL_SIZE)

    def move(self):
        self.x += self.dx
        self.y += self.dy
        self.rect.x = self.x - BALL_SIZE // 2
        self.rect.y = self.y - BALL_SIZE // 2

        # Bounce off top and bottom
        if self.y <= 0 or self.y >= WINDOW_HEIGHT:
            self.dy = -self.dy

        # Bounce off right wall
        if self.x >= WINDOW_WIDTH:
            self.dx = -self.dx

        # Reset if ball passes left wall
        if self.x < 0:
            self.reset()

# Game setup
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Pong Game (Single Player)")
clock = pygame.time.Clock()

paddle = Paddle()
ball = Ball()

def setup():
    global paddle, ball
    screen.fill(BLACK)
    paddle = Paddle()
    ball = Ball()
    pygame.mouse.set_visible(False)  # Hide mouse cursor

def draw():
    screen.fill(BLACK)
    # Draw paddle
    pygame.draw.rect(screen, WHITE, paddle.rect)
    # Draw ball
    pygame.draw.rect(screen, WHITE, ball.rect)
    pygame.display.flip()

def update_loop():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.mouse.set_visible(True)  # Restore mouse cursor on quit
            pygame.quit()
            return

    # Paddle movement based on mouse position
    mouse_y = pygame.mouse.get_pos()[1]
    paddle.move(mouse_y)

    # Ball movement
    ball.move()

    # Ball-paddle collision
    if paddle.rect.colliderect(ball.rect):
        ball.dx = -ball.dx  # Reverse horizontal direction
        ball.dx *= 1.1  # Slight speed increase

    draw()
    clock.tick(FPS)

async def main():
    setup()
    while True:
        update_loop()
        await asyncio.sleep(1.0 / FPS)

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())