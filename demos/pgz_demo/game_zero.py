import random
import pgzrun
from pgzero.actor import Actor

# Window dimensions
WIDTH  = 800
HEIGHT = 600

# 1) Player using built-in 'ship' sprite
player = Actor("ship")
player.pos = (WIDTH // 2, HEIGHT // 2)

# 2) Enemies using built-in 'alien' sprite
enemies = []
for _ in range(5):
    x = random.randint(50, WIDTH - 50)
    y = random.randint(50, HEIGHT - 50)
    enemy = Actor("alien", center=(x, y))
    enemies.append(enemy)

# 3) Update loop: movement and simple AI
def update():
    # Player movement (WASD + arrows)
    if keyboard.left  or keyboard.a: player.x -= 5
    if keyboard.right or keyboard.d: player.x += 5
    if keyboard.up    or keyboard.w: player.y -= 5
    if keyboard.down  or keyboard.s: player.y += 5

    # Keep player on-screen
    player.x = max(0, min(WIDTH,  player.x))
    player.y = max(0, min(HEIGHT, player.y))

    # Enemies random jitter
    for e in enemies:
        e.x += random.uniform(-2, 2)
        e.y += random.uniform(-2, 2)
        e.x = max(0, min(WIDTH,  e.x))
        e.y = max(0, min(HEIGHT, e.y))

# 4) Draw loop: clear and draw actors
def draw():
    screen.clear()
    player.draw()
    for e in enemies:
        e.draw()

# 5) Launch Pygame Zero
pgzrun.go()
