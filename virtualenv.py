from concurrent.futures import ThreadPoolExecutor
import random
from typing import List, Literal, Tuple
import numpy as np
from agent import Agent
from noise import snoise2, pnoise2

class Environment:
    def __init__(self, width: int, height: int, population: int = 100):
        self.width = width
        self.height = height
        self.population = population

        self.environment = self.generate_environment()
        self.foods = self.generate_foods(100)
        self.agents = self.generate_agents(population)

        self.max_infection = None
        self.max_temperature = None

        self.step_count = 0

    def generate_environment(self, scale=10.0, octaves=7, persistence=1.1, lacunarity=1.1, seed=None):
        if seed is None:
            seed = random.randint(0, 10000)
            random.seed(seed)
        else:
            random.seed(seed)

        environment = np.zeros((self.height, self.width, 3), dtype=np.float16)

        x_add = random.randint(-10000, 10000)
        y_add = random.randint(-10000, 10000)

        for y in range(self.height):
            for x in range(self.width):
                nx = (x / scale) + x_add
                ny = (y / scale) + y_add

                block_value = snoise2(nx, ny, octaves=octaves, persistence=1.1, lacunarity=1.1, base=seed)
                environment[y, x, 0] = np.float16(1 if block_value >= 0.01 else 0) if x != 0 and y != 0 and x != self.width - 1 and y != self.height - 1 else 1

                infection_value = pnoise2(nx, ny, octaves=octaves, persistence=persistence, lacunarity=lacunarity, base=seed + 1)
                environment[y, x, 1] = np.float16(max(0, infection_value))

                temp_value = (pnoise2(nx, ny, octaves=octaves, persistence=persistence, lacunarity=lacunarity, base=seed + 2) * 2) - 1
                environment[y, x, 2] = np.float16(temp_value)

        return environment

    def generate_foods(self, count: int) -> List[Tuple[int, int]]:
        foods = []
        empty_cells = np.argwhere(self.environment[:, :, 0] == 0)
        chosen_indices = np.random.choice(len(empty_cells), count, replace=False)
        for idx in chosen_indices:
            y, x = empty_cells[idx]
            self.environment[y, x, 0] = 4
            foods.append((x, y))
        return foods

    def delete_all_foods(self) -> int:
        for x, y in self.foods:
            self.environment[y, x, 0] = 0
        count = len(self.foods)
        self.foods = []
        return count

    def reset(self, food_count: int = 100):
        self.environment = self.generate_environment()
        self.delete_all_foods()
        self.foods = self.generate_foods(food_count)
        # self.agents = self.generate_agents(self.population)
        self.max_infection = None
        self.max_temperature = None
        for agent in self.agents:
            agent.reset()

    def generate_agents(self, count: int) -> list[Agent]:
        agents = []
        for _ in range(count):
            x, y = 0, 0
            while self.environment[y, x, 0] != 0:
                x = random.randint(0, self.width - 1)
                y = random.randint(0, self.height - 1)
            agents.append(Agent((x, y), state_dim=81, action_dim=4))
        return agents

    def agents_done(self, procent: float) -> bool:
        return len(self.agents) * procent <= sum(agent.done for agent in self.agents)
    
    def get_avg_epsilon(self) -> float:
        return sum(agent.epsilon for agent in self.agents) / len(self.agents)

    def step(self):
        for agent in self.agents:
            if agent.done:
                continue
            state = self.get_state(*agent.position)
            state = np.concatenate([state, [agent.hp, agent.hunger, agent.infection, agent.temperature]])

            action = agent.act(state)
            new_position, normal_new_position = self.get_new_position(agent.position, action)

            if self.environment[normal_new_position[1], normal_new_position[0], 0] == 0:
                agent.position = new_position
                agent.reward -= 0.1
            elif self.environment[normal_new_position[1], normal_new_position[0], 0] == 1:
                agent.reward -= .5
            elif self.environment[normal_new_position[1], normal_new_position[0], 0] == 4:
                agent.reward += 5
                agent.hunger = 10
                self.environment[normal_new_position[1], normal_new_position[0], 0] = 0
                self.foods.remove(normal_new_position)
                agent.position = new_position

            agent.infection += self.environment[normal_new_position[1], normal_new_position[0], 1]
            agent.temperature += self.environment[normal_new_position[1], normal_new_position[0], 2]
            agent.hunger -= 1
            
            if agent.hunger <= 0:
                agent.hp -= 1
                agent.hunger = 0
                agent.reward -= .1
            if agent.hp <= 0:
                agent.done = True

            next_state = self.get_state(*agent.position)
            next_state = np.concatenate([next_state, [agent.hp, agent.hunger, agent.infection, agent.temperature]])
            agent.remember(state, action, agent.reward, next_state, agent.done)

        if self.step_count % 100 == 0:
            # # Многопоточное обучение всех агентов
            with ThreadPoolExecutor(max_workers=56) as executor:
                executor.map(lambda agent: agent.replay(32), self.agents)

            # for agent in self.agents:
            #     agent.replay(32)
            
            # Обновление целевой сети
            for agent in self.agents:
                agent.update_target_model()

        self.step_count += 1
        return self.get_max_reward(), self.get_top_agents(10)

    def get_new_position(self, position: tuple[int, int], action: Literal[0, 1, 2, 3]) -> tuple[tuple[int, int], tuple[int, int]]:
        x, y = position
        if action == 0:
            y -= 1
        elif action == 1:
            y += 1
        elif action == 2:
            x -= 1
        elif action == 3:
            x += 1
        normal_x, normal_y = x % self.width, y % self.height
        return (x, y), (normal_x, normal_y)

    def get_state(self, x, y):
        neighbors = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                neighbors.append(self.environment[(y + dy) % self.height, (x + dx) % self.width])

        center = self.environment[y % self.height, x % self.width]
        return np.concatenate([center] + neighbors + [[x, y]])

    def get_reward(self, agent: Agent) -> float:
        if self.max_infection is None:
            self.max_infection = max(agent.infection for agent in self.agents)
        if self.max_temperature is None:
            self.max_temperature = max(abs(agent.temperature) for agent in self.agents)

        agent_infection = agent.infection / (self.max_infection + 1)
        agent_temperature = abs(agent.temperature) / (self.max_temperature + 1)

        infection_penalty = 1 - agent_infection
        temperature_penalty = 1 - agent_temperature

        reward = agent.reward * infection_penalty * temperature_penalty

        return reward

    def get_max_reward(self) -> tuple[Agent, float]:
        max_reward = max(self.agents, key=lambda agent: agent.reward)
        return max_reward, max_reward.reward

    def get_top_agents(self, count: int) -> list[Agent]:
        return sorted(self.agents, key=lambda agent: self.get_reward(agent), reverse=True)[:count]

if __name__ == "__main__":
    env = Environment(100, 100, 50)
    profiler = cProfile.Profile()
    profiler.enable()
    env.reset()
    profiler.disable()
    profiler.print_stats(sort='time')