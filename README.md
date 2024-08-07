## Structure

### Python

The primary purpose of the Python code is to integrate a Java-based simulation into a reinforcement learning (RL) environment. This integration enables the RL agent to interact with and learn from a dynamic environment modeled by the Java simulation. Here's a detailed breakdown of the process:

1. Java Simulation Interface: The Python code interfaces with the Java simulation using JPype, allowing it to execute Java methods directly. This setup is crucial for accessing the real-time simulation data necessary for RL training.
2. Python Simulation Interface: For more convenience, there is also a Python simulation that runs without Java. It is a simplified version fo the Java simulation and is satisfying for  the scope of this thesis. The output of both simulations, a graph instance, is identical making the interchange between the two possible
3. Creation of the RL Environment:
Within this environment, Python leverages the results from periodic simulation intervals provided by the Java/Python backend. Each interval yields data that Python transforms into the current state of the environment, represented as a graph structure.
4. Graph Construction:
The state graph is constructed using data such as node interactions, connections, and other relevant metrics from the simulation output. This graph forms the basis of the environment state at each step in the simulation.
5. State to Reward Mapping:
The environment processes the state graph to compute rewards. These rewards are based on predefined criteria such as performance metrics or achievement of specific objectives within the simulation, which are pivotal for training the RL agent effectively.
6. Observation Wrapper:
An observation wrapper is employed to adapt the state representation into a format suitable for processing by graph neural networks (GNNs). This wrapper converts the state graph into a PyTorch Geometric graph, which is a compatible input for GNNs.
7. Utilization of Graph Neural Networks:
GNNs serve as the function approximators in the RL agent. They process the graph-based state representations to make decisions or predict the next actions. Their capability to handle graph data makes them ideal for environments where the state is expressed as interconnected nodes and edges.
8. RL Agent Training:
With the environment set up and the observation wrapper in place, the RL agent undergoes training. It interacts with the environment, receiving states and rewards, and learns to optimize its policy based on the feedback received through the GNN's predictions.
9. Continuous Learning and Adaptation:
As the simulation progresses, the Python environment continually updates the state based on new data from the Java/Python simulation, allowing the RL agent to adapt to changing conditions and potentially complex dynamics modeled by the simulation.
10. Summary
This system exemplifies a robust integration of a Java-based simulation with a Python-managed RL environment, leveraging the strengths of both platforms. The use of GNNs as function approximators enhances the agent's ability to interpret complex, graph-based state inputs, facilitating sophisticated decision-making processes in environments modeled by simulations.

- `env` directory contains the code to create the environment using the java simulation code
  - `env.py` is the main file that contains the environment class and some abstract classes
  - `java.py` is the file that contains the code to run the java simulation code and get the result of the simulation
  - `conversion_utils.py` is the file that contains the code to convert the result of the simulation to a graph
- `model` directory contains the code to create the Graph Neural Network
  - `gnn.py` is the file that contains the code to create the Graph Neural Network
  - `temp_gnn.py` is the file containing trial spatio-temporal GNNs 
- `algorithm` directory contains the code for the specific RL algorithms of DQN (Double DQN), PPO and experimental imitation training on optimal relocation patterns
  - `agents.py` contains the actual implementation of PPO and DQN from scratch
  - `memory.py` is the file for experience replay buffer or on policy memory
  - `supervised.py` is the file about the experimental supervised training of predicting the aggregated latency
  - `trainer.py` is the file running the training algorithm by interacting with the environment and using an agent instance
- `env` directory contains the code for the creating of the python simulation of the replicated state machine system and the connection to the Java backend which also runs a simulation if needed
  - `conversion_utils.py` contains a utility function that converts a dictionary of latencies to a graph
  - `data_collector.py` is the file for collecting optimal relocation tuples by having a human select the best configuration using a dashboard. This lets us create expert samples for imitation learning
  - `env.py` is the file containing all relevant OpenAI gym environments interface implementations as well as observation wrappers converting raw graphs into PyTorch graphs
  - `java.py` has all the relevant code to connect Python to Java
  - `network_simulation.py` contains the simulation in Python, all its structure and logic are implemented here. This file is the foundation for all the OpenAI gym interface implementation. It runs simulation intervals, can render the environment, and computes rewards as well as stochastic client movements internally.
### Java


Currently you can find a mockup of the Java project structure in the `java` directory which mocks a simulation of graph based network 

- `simulation` directory contains the simulation code
  - `src` directory contains the source code
    - `main` directory contains the main code
      - `MockSimulation.java` is the main class used to run the simulation
      - `IntervalRequest.java` is the class that represents the network state after each interval
