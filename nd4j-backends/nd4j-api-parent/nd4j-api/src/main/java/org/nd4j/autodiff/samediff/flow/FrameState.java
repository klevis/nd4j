package org.nd4j.autodiff.samediff.flow;

import lombok.Data;
import lombok.NonNull;

/**
 * This class is a holder for state of loops imported from TensorFlow, via frame_name
 *
 * @author raver119@gmail.com
 */
@Data
public class FrameState {
    private String name;
    private long iterations = 0;
    private boolean rewindPlanned = false;
    private int rewindPosition = -1;


    public FrameState(@NonNull String frame_name) {
        this.name = frame_name;
    }
}
