# Virtual Environment Simulation

This project is a test implementation of a virtual environment simulation using Python. The code is written with the help of GitHub Copilot and is not guaranteed to be fully functional or optimized.

## Features

- **Virtual Environment**: A simulated environment where agents interact with their surroundings.
- **Agents**: Autonomous agents that can move, eat, and interact with the environment.
- **Graphical Interface**: Visualization of the environment and agents using Pygame.
- **Machine Learning**: Agents use a Deep Q-Network (DQN) for decision making.

## Requirements

The project requires the following Python packages:

- `pygame`
- `numpy`
- `scipy`
- `torch`
- `noise`

You can install these dependencies using the following command:

```sh
pip install -r requirements.txt
```

## Usage

To run the simulation, execute the `main.py` file:

```sh
python main.py
```

## Project Structure

- `main.py`: Main script to run the simulation.
- `agent.py`: Contains the `Agent` class which defines the behavior of the agents.
- `model.py`: Defines the DQN model used by the agents.
- `virtualenv.py`: Contains the `Environment` class which defines the virtual environment.
- `requirements.txt`: List of required Python packages.
- `.gitignore`: Specifies files and directories to be ignored by Git.
- `README.md`: This file.

## Notes

- This project is a test implementation and may not be fully functional.
- The code is generated with the help of GitHub Copilot.

## License

This project is licensed under the MIT License.