from typing import List, Tuple
import pygame
import sys
import numpy as np
from virtualenv import Environment
from scipy.interpolate import make_interp_spline

# Инициализация Pygame
pygame.init()

# Размеры окна
WIDTH, HEIGHT = 1600, 800
CELL_SIZE = 10

env = Environment(WIDTH // CELL_SIZE, HEIGHT // CELL_SIZE)
# env.generate_agents_from_loaded("models/best_agent_460.75.pth", 100)

# Создание окна
screen = pygame.display.set_mode((WIDTH, HEIGHT+200))
pygame.display.set_caption("Virtual Environment")

# Цвета
COLORS = {
    0: (50, 50, 50),  # Пустая клетка - белый
    1: (0, 0, 0),        # Блок - черный
    4: (0, 255, 0)       # Еда - зеленый
}

MIN_FOOD = 500
MAX_FOOD = 500
FOOD_GROWTH = 0

food_count = MIN_FOOD
# Создание среды
max_value = int(-1e9)
min_value = int(1e9)

def draw_graph(screen: pygame.Surface, data: List[int], x_start: int, y_start: int, width: int, height: int, color: Tuple[int, int, int] = (0, 0, 255)) -> None:
    if len(data) < 2:
        return

    global max_value, min_value

    max_value = max(max(data), max_value)
    min_value = min(min(data), min_value)
    data_range = max_value - min_value if max_value != min_value else 1

    points = []
    for i in range(len(data)):
        x = x_start + i * (width / (len(data) - 1))
        y = y_start + height - (data[i] - min_value) * height / data_range

        # Проверка на переполнение и корректировка значений
        if not np.isfinite([x, y]).all():
            continue

        # Округление координат
        x, y = int(x), int(y)

        pygame.draw.line(screen, (0, 0, 0), (x, y_start), (x, y_start + height - 1), 1)
        pygame.draw.line(screen, (0, 0, 0), (x_start, y), (x_start + width - 1, y), 1)
        
        pygame.draw.circle(screen, (255, 0, 0), (x, y), 2.5)

        # Проверка координат
        if 0 <= x <= screen.get_width() and 0 <= y <= screen.get_height():
            points.append((x, y))

    if len(points) > 3:
        points = np.array(points)
        x_points = points[:, 0]
        y_points = points[:, 1]

        # Интерполяция сплайнами
        x_new = np.linspace(x_points.min(), x_points.max(), 300)
        spl = make_interp_spline(x_points, y_points, k=2)
        y_new = spl(x_new)

        smooth_points = list(zip(x_new, y_new))
        pygame.draw.lines(screen, color, False, smooth_points, 3)
    # pygame.draw.lines(screen, (255, 0, 0), False, points, 1)
            

def draw_environment():
    for y in range(env.height):
        for x in range(env.width):
            cell_value = env.environment[y, x, 0]
            color = COLORS.get(cell_value, (255, 0, 0))  # По умолчанию - красный
            pygame.draw.rect(screen, color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

def draw_agents():
    for agent in env.agents:
        pygame.draw.circle(screen, (0, 0, 255), (agent.position[0] * CELL_SIZE + CELL_SIZE // 2, agent.position[1] * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 2)

# Основной цикл
running = True
clock = pygame.time.Clock()
font = pygame.font.Font(None, 18)
best_agent = None

rewards = []

render = True
delta_time = 0
epsion = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                env.reset()
            if event.key == pygame.K_r:
                render = not render
            if event.key == pygame.K_s:
                top_10_agent[0].model.save(f"models/best_agent_{best_agent.reward}.pth")
                

    start_time_env = pygame.time.get_ticks()
    if not render:
    # Обновление среды
        while not env.agents_done(.9):
            max_agent_and_max_reward, top_10_agent = env.step()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
    else:
        max_agent_and_max_reward, top_10_agent = env.step()

    max_agent, max_reward = max_agent_and_max_reward

    # Отрисовка
    screen.fill((50, 50, 50))
    if render:
        draw_environment()
        draw_agents()
        draw_graph(screen, rewards, 0, HEIGHT, WIDTH, 200)
    else:
        draw_graph(screen, rewards, 0, 20, WIDTH, HEIGHT, (0, 0, 255))
    # Отрисовка графиков
    

    # Вывод информации о лучшем агенте и топ-10 агентов
    # pygame.draw.rect(screen, (255, 255, 255), (0, 0, WIDTH, CELL_SIZE * 2), border_radius=10)
    text = font.render(f"Max agent reward: {max_reward}", True, (128, 0, 128))
    screen.blit(text, (10, HEIGHT))
    for i, agent in enumerate(top_10_agent):
        text = font.render(f"Agent {i + 1}: {agent.reward}", True, (128, 0, 128))
        screen.blit(text, (10, HEIGHT + (i + 1) * 20))

    pygame.display.flip()

    if env.agents_done(.9):
        if best_agent is None or best_agent.reward < top_10_agent[0].reward:
            best_agent = top_10_agent[0]
            print(f"Best agent reward: {best_agent.reward}")
            # best_agent.model.save(f"models/best_agent_{best_agent.reward}.pth")
        if len(rewards) > 50:
            rewards = rewards[1:]
        rewards.append(top_10_agent[0].reward)
        food_count = max(MIN_FOOD, food_count - FOOD_GROWTH)
        start_time = pygame.time.get_ticks()
        env.reset(food_count)
        delta_time = pygame.time.get_ticks() - start_time
        # print(f"Time: {pygame.time.get_ticks() - start_time} ms")
        epsion = best_agent.epsilon

    pygame.display.set_caption(f"VE {clock.get_fps():.0f} Food: {food_count} T:{delta_time}ms TE:{pygame.time.get_ticks() - start_time_env}ms E:{env.get_avg_epsilon():.2f}")
    # Ограничение FPS
    clock.tick(60)

pygame.quit()
sys.exit()