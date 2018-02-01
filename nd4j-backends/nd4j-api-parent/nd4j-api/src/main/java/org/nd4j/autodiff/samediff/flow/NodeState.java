package org.nd4j.autodiff.samediff.flow;

import lombok.Data;
import lombok.NonNull;

/**
 * This class describe Node state during execution time.
 *
 * @author raver119@gmail.com
 */
@Data
public class NodeState {
    private String nodeName;
    private boolean active = true;
    private int activeBranch = 0;
    private boolean executed = false;

    public NodeState(@NonNull String nodeName) {
        this.nodeName = nodeName;
    }
}
