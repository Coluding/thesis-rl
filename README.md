# Thesis Repository
## Title:
Optimizing Allocation Location Problems: A Hybrid Approach with Graph Neural Networks andDeep Reinforcement Learning


## Abstract:
This thesis explores the optimization of dynamic Location Allocation (LA) problems with a focus on Replicated State Machines (RSM) in large-scale global networks. RSM systems, which ensure fault tolerance through the replication of state machines across multiple data centers, face significant latency challenges due to global client distribution and the need for dynamic reconfiguration of active data centers. Traditional heuristic approaches are limited in their scalability and ability to adapt to rapidly changing environments.\\
To address these challenges, this research proposes a hybrid optimization framework that leverages Graph Neural Networks (GNNs) and Deep Reinforcement Learning (DRL). GNNs are employed to capture the complex graph structure of RSM systems, representing data centers and client interactions. DRL is utilized to train the model in an online learning environment, enabling it to dynamically adjust data center configurations to minimize latency and operational costs.\\
The thesis first provides a comprehensive review of LA problems and their classification, followed by an exploration of classical and modern heuristic solution methods, including the theoretical foundation of DRL and GNNs. A novel application of the Dynamic Stochastic Facility Location Problem (DSFLP) is developed to frame the RSM optimization challenge within an RL context. The proposed solution is evaluated in a simulated environment, demonstrating its effectiveness in reducing latency and improving system performance compared to traditional methods.\\
This work contributes to the field of operations research by integrating advanced machine learning techniques with classical optimization problems, offering a scalable and adaptive solution for real-world applications in global network management.


## Structure
It contains Python and Java Code. The Java Code is just a mockup to test the connection to the Python code. The goal is to build a DRL algorithm that solves the optimal relocation challenge of the replicated state machine system.  Everything is implemented using OpenAI Gym and PyTorch to build a modular and reusable code base.
### Python

The primary purpose of the Python code is to integrate a Java-based simulation into a reinforcement learning (RL) environment. This integration enables the RL agent to interact with and learn from a dynamic environment modeled by the Java simulation. Here's a detailed breakdown of the process:

1. Java Simulation Interface: The Python code interfaces with the Java simulation using JPype, allowing it to execute Java methods directly. This setup is crucial for accessing the real-time simulation data necessary for RL training.
2. Python Simulation Interface: For more convenience, there is also a Python simulation that runs without Java. It is a simplified version of the Java simulation and is satisfying for  the scope of this thesis. The output of both simulations, a graph instance, is identical making the interchange between the two possible
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
