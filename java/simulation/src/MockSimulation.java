import java.util.*;

public class MockSimulation {

    private int numNodes;
    private int numActive;
    private int numPassive;
    private int numOff;
    private int numClients; // Number of client nodes
    private Map<Integer, Map<Integer, Integer>> currentSystem;
    private Map<String, List<Integer>> currentSystemConfiguration;

    public MockSimulation(int numNodes, int numActive, int numPassive, int numClients){
        this.numNodes = numNodes;
        this.numActive = numActive;
        this.numPassive = numPassive;
        this.numClients = numClients;
        this.numOff = numNodes - numActive - numPassive - numClients;
        this.currentSystemConfiguration = new HashMap<>();
        initializeNodes();
    }

    private void initializeNodes() {
        // List to hold all node IDs
        List<Integer> nodeIds = new ArrayList<>();
        for (int i = 0; i < numNodes; i++) {
            nodeIds.add(i);
        }

        // Shuffle the list to randomize node distribution
        Collections.shuffle(nodeIds);

        // Distribute nodes into different states including clients
        currentSystemConfiguration.put("active", nodeIds.subList(0, numActive));
        currentSystemConfiguration.put("passive", nodeIds.subList(numActive, numActive + numPassive));
        currentSystemConfiguration.put("off", nodeIds.subList(numActive + numPassive, numNodes - numClients));
        currentSystemConfiguration.put("clients", nodeIds.subList(numNodes - numClients, numNodes));
    }

    public Map<String, List<Integer>> getCurrentSystemConfiguration() {
        return currentSystemConfiguration;
    }

    public IntervalResult runInterval() {

        Map<Integer, Map<Integer, Integer>> outerMap = new HashMap<>();
        Random random = new Random();

        // Initialize maps for all nodes
        for (int i = 0; i < numNodes; i++) {
            outerMap.put(i, new HashMap<>());
        }

        // Connect each client to every active node with random latency
        List<Integer> clientNodes = currentSystemConfiguration.get("clients");
        List<Integer> activeNodes = currentSystemConfiguration.get("active");
        for (int client : clientNodes) {
            Map<Integer, Integer> connections = outerMap.get(client);
            for (int active : activeNodes) {
                connections.put(active, random.nextInt(100)); // Random latency up to 100
            }
        }

        // Other random connections (as before)
        for (int i = 0; i < numNodes; i++) {
            Map<Integer, Integer> innerMap = outerMap.get(i);

            if (random.nextBoolean()) {
                for (int j = 0; j < numNodes; j++) {
                    if (i != j && random.nextDouble() < 0.20) {
                        innerMap.put(j, random.nextInt(100));
                    }
                }
            }
        }

        this.currentSystem = outerMap;
        return new IntervalResult(outerMap, currentSystemConfiguration);
    }

    public void setPlacement(int action) {
        // Retrieve current configurations and create independent lists
        List<Integer> activeNodes = new ArrayList<>(currentSystemConfiguration.get("active"));
        List<Integer> passiveNodes = new ArrayList<>(currentSystemConfiguration.get("passive"));
        List<Integer> offNodes = new ArrayList<>(currentSystemConfiguration.get("off"));
        List<Integer> clientNodes = currentSystemConfiguration.get("clients");

        // Validate node index
        if (action < 0 || action >= numNodes) {
            throw new IllegalArgumentException("Invalid node index.");
        }

        // Ensure the action node is not a client node
        if (clientNodes.contains(action)) {
            throw new IllegalArgumentException("Client nodes cannot be set as passive.");
        }

        // Find the currently passive node
        if (passiveNodes.isEmpty()) {
            throw new IllegalStateException("No passive node is currently set.");
        }
        int currentPassive = passiveNodes.get(0);  // Assuming there is exactly one passive node

        // Check if the action node is already passive
        if (currentPassive == action) {
            System.out.println("Node " + action + " is already passive.");
            return;
        }

        // Identify and switch states
        passiveNodes.clear();  // Clear the passive list first
        passiveNodes.add(action);  // Make the action node passive

        // Switch the previously passive node to the new state of the action node
        if (activeNodes.contains(action)) {
            activeNodes.remove(Integer.valueOf(action));
            activeNodes.add(currentPassive);  // Swap passive to active
        } else if (offNodes.contains(action)) {
            offNodes.remove(Integer.valueOf(action));
            offNodes.add(currentPassive);  // Swap passive to off
        } else {
            throw new IllegalStateException("Node " + action + " is in an unknown state.");
        }

        // Update the current configuration
        currentSystemConfiguration.put("active", activeNodes);
        currentSystemConfiguration.put("passive", passiveNodes);
        currentSystemConfiguration.put("off", offNodes);
    }

    public IntervalResult reset() {
        Map<Integer, Map<Integer, Integer>> outerMap = new HashMap<>();

        // Randomly decide the number of edges each node might have (0 to numNodes-1)
        for (int i = 0; i < numNodes; i++) {
            Map<Integer, Integer> innerMap = new HashMap<>();
            outerMap.put(i, innerMap);
        }

        this.currentSystem = outerMap;

        return new IntervalResult(outerMap, currentSystemConfiguration);
    }

    public static void main(String[] args) {
        System.out.println(System.getProperty("java.class.path"));
        MockSimulation obj = new MockSimulation(12,4, 1, 2);

        System.out.println(obj.runInterval().toString());
        obj.setPlacement(3);
        System.out.println(obj.getCurrentSystemConfiguration().toString());
    }
}
