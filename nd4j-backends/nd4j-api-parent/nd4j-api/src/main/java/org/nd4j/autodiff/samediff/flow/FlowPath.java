package org.nd4j.autodiff.samediff.flow;

import lombok.NonNull;

import java.util.HashMap;
import java.util.Map;

/**
 * This class acts as holder for flow control information.
 *
 * @author raver119@gmail.com
 */
public class FlowPath {
    protected Map<String, NodeState> states = new HashMap<>();


    public void ensureNodeStateExists(@NonNull String nodeName) {
        if (!states.containsKey(nodeName))
            states.put(nodeName, new NodeState(nodeName));
    }

    public boolean isActive(@NonNull String nodeName) {
        ensureNodeStateExists(nodeName);

        return states.get(nodeName).isActive();
    }

    public void markActive(@NonNull String nodeName, boolean active) {
        ensureNodeStateExists(nodeName);

        states.get(nodeName).setActive(active);
    }

    public void setActiveBranch(@NonNull String nodeName, int branchIdx) {
        states.get(nodeName).setActiveBranch(branchIdx);
    }

    public int getActiveBranch(@NonNull String nodeName) {
        return states.get(nodeName).getActiveBranch();
    }

    public boolean wasExecuted(@NonNull String nodeName) {
        ensureNodeStateExists(nodeName);

        return states.get(nodeName).isExecuted();
    }

    public void markExecuted(@NonNull String nodeName, boolean executed) {

        states.get(nodeName).setExecuted(executed);
    }

    public void incrementNumberOfCycles(@NonNull String nodeName) {
        states.get(nodeName).incrementNumberOfCycles();
    }

    public long getNumberOfCycles(@NonNull String nodeName) {
        return states.get(nodeName).getNumberOfCycles();
    }

}
