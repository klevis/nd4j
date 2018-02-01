package org.nd4j.autodiff.samediff.flow;

import lombok.NonNull;

import java.util.Map;

/**
 * This class acts as holder for flow control information.
 *
 * @author raver119@gmail.com
 */
public class FlowPath {
    protected Map<String, NodeState> states;


    protected void ensureNodeStateExists(@NonNull String nodeName) {
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
        ensureNodeStateExists(nodeName);

        states.get(nodeName).setActiveBranch(branchIdx);
    }

    public int getActiveBranch(@NonNull String nodeName) {
        ensureNodeStateExists(nodeName);

        return states.get(nodeName).getActiveBranch();
    }

}
