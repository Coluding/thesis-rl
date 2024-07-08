import java.util.List;
import java.util.Map;

public class IntervalResult {
    private Map<Integer, Map<Integer, Integer>> currentSystemLatencies;
    private Map<String, List<Integer>> currentSystemConfiguration;

    public IntervalResult(Map<Integer, Map<Integer, Integer>> currentSystemLatencies,
                          Map<String, List<Integer>> currentSystemConfiguration) {
        this.currentSystemLatencies = currentSystemLatencies;
        this.currentSystemConfiguration = currentSystemConfiguration;
    }

    public Map<Integer, Map<Integer, Integer>> getCurrentSystemLatencies() {
        return currentSystemLatencies;
    }

    public Map<String, List<Integer>> getCurrentSystemConfiguration() {
        return currentSystemConfiguration;
    }

    @Override
    public String toString() {
        return "IntervalResult{" +
                "currentSystemLatencies=" + currentSystemLatencies +
                ", currentSystemConfiguration=" + currentSystemConfiguration +
                '}';
    }
}