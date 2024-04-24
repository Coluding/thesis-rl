## Structure

### Java

Currently you can find a mockup of the Java project structure in the `java` directory which mocks a simulation of graph based network 

- `simulation` directory contains the simulation code
  - `src` directory contains the source code
    - `main` directory contains the main code
      - `MockSimulation.java` is the main class used to run the simulation
      - `IntervalRequest.java` is the class that represents the network state after each interval


### Python

The python code is the main code that is used to integrate the java simulation code with the python code. The python code is used to create an RL environment.
In the environment the result of the simulation interval is used to create the state of the environment as a graph. The graph is then used to create the state of the environment. The state is then used to create the reward and the next state of the environment. The environment is then used to train the RL agent.
I added an observation wrapper to the environment to return the state of the environment as a pytorch geometric graph. The usae of Graph Neural Networks is then possible as they are the function approximators used in the RL agent.

- `env` directory contains the code to create the environment using the java simulation code
  - `env.py` is the main file that contains the environment class and some abstract classes
  - `java.py` is the file that contains the code to run the java simulation code and get the result of the simulation
  - `conversion_utils.py` is the file that contains the code to convert the result of the simulation to a graph
- `model` directory contains the code to create the Graph Neural Network
  - `gnn.py` is the file that contains the code to create the Graph Neural Network