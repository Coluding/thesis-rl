import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class MockSimulation {

    private int numNodes;
    private Map<Integer, Map<Integer, Integer>> currentSystem;

    public MockSimulation(int numNodes){
        this.numNodes = numNodes;
    }

    public void setNumNodes(int numNodes) {
        this.numNodes = numNodes;
    }

    public Map<Integer, Map<Integer, Integer>> runInterval() {

        Map<Integer, Map<Integer, Integer>> outerMap = new HashMap<>();
        Random random = new Random();

        // Randomly decide the number of edges each node might have (0 to numNodes-1)
        for (int i = 1; i <= numNodes; i++) {
            Map<Integer, Integer> innerMap = new HashMap<>();

            if (random.nextBoolean()) { // Randomly decide to skip some nodes
                for (int j = 1; j <= numNodes; j++) {
                    if (i != j && random.nextBoolean()) { // Ensure no self-loop and not every connection is made
                        innerMap.put(j, random.nextInt(100)); // Random values up to 100
                    }
                }
            }

            outerMap.put(i, innerMap);
        }

        this.currentSystem = outerMap;
        return outerMap;
    }

    public void setPlacement(int action){
        System.out.println("set placement");
    }

    public Map<Integer, Map<Integer, Integer>> reset(){
        return this.runInterval();
    }

    public static void main(String[] args) {
        System.out.println(System.getProperty("java.class.path"));
        MockSimulation obj = new MockSimulation(10);
        System.out.println(obj.runInterval().toString());
    }
}
