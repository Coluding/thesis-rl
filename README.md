## Structure

### Java

Currently you can find a mockup of the Java project structure in the `java` directory which mocks a simulation of graph based network 

- `simulation` directory contains the simulation code
  - `src` directory contains the source code
    - `main` directory contains the main code
      - `MockSimulation.java` is the main class used to run the simulation
      - `IntervalRequest.java` is the class that represents the network state after each interval


### Python

The primary purpose of the Python code is to integrate a Java-based simulation into a reinforcement learning (RL) environment. This integration enables the RL agent to interact with and learn from a dynamic environment modeled by the Java simulation. Here's a detailed breakdown of the process:

1. Java Simulation Interface: The Python code interfaces with the Java simulation using JPype, allowing it to execute Java methods directly. This setup is crucial for accessing the real-time simulation data necessary for RL training.
2. Creation of the RL Environment:
Within this environment, Python leverages the results from periodic simulation intervals provided by the Java backend. Each interval yields data that Python transforms into the current state of the environment, represented as a graph structure.
3. Graph Construction:
The state graph is constructed using data such as node interactions, connections, and other relevant metrics from the simulation output. This graph forms the basis of the environment state at each step in the simulation.
4. State to Reward Mapping:
The environment processes the state graph to compute rewards. These rewards are based on predefined criteria such as performance metrics or achievement of specific objectives within the simulation, which are pivotal for training the RL agent effectively.
5. Observation Wrapper:
An observation wrapper is employed to adapt the state representation into a format suitable for processing by graph neural networks (GNNs). This wrapper converts the state graph into a PyTorch Geometric graph, which is a compatible input for GNNs.
6. Utilization of Graph Neural Networks:
GNNs serve as the function approximators in the RL agent. They process the graph-based state representations to make decisions or predict the next actions. Their capability to handle graph data makes them ideal for environments where the state is expressed as interconnected nodes and edges.
7. RL Agent Training:
With the environment set up and the observation wrapper in place, the RL agent undergoes training. It interacts with the environment, receiving states and rewards, and learns to optimize its policy based on the feedback received through the GNN's predictions.
8. Continuous Learning and Adaptation:
As the simulation progresses, the Python environment continually updates the state based on new data from the Java simulation, allowing the RL agent to adapt to changing conditions and potentially complex dynamics modeled by the simulation.
9. Summary
This system exemplifies a robust integration of a Java-based simulation with a Python-managed RL environment, leveraging the strengths of both platforms. The use of GNNs as function approximators enhances the agent's ability to interpret complex, graph-based state inputs, facilitating sophisticated decision-making processes in environments modeled by simulations.

- `env` directory contains the code to create the environment using the java simulation code
  - `env.py` is the main file that contains the environment class and some abstract classes
  - `java.py` is the file that contains the code to run the java simulation code and get the result of the simulation
  - `conversion_utils.py` is the file that contains the code to convert the result of the simulation to a graph
- `model` directory contains the code to create the Graph Neural Network
  - `gnn.py` is the file that contains the code to create the Graph Neural Network