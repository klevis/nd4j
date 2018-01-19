package org.nd4j.linalg.api.ops.impl.accum;

import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Cumulative sum operation, optionally along dimension.
 *
 * @author Alex Black
 */
public class CumSum extends DynamicCustomOp {

    public CumSum(){

    }

    public CumSum(SameDiff sameDiff, SDVariable x, int... dimension){
        super(null, sameDiff, new SDVariable[]{x});
        this.dimensions = dimension;
        addIArgument(dimension);
    }

    @Override
    public String opName() {
        return "cumsum";
    }

    @Override
    public String tensorflowName() {
        return "Cumsum";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        super.initFromTensorFlow(nodeDef, initWith, attributesForNode, graph);
    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        super.initFromOnnx(node, initWith, attributesForNode, graph);
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad){
        //Gradient should be cumulative sum along same dimension of the *reversed* array

        SDVariable reverse = f().reverse(grad.get(0), dimensions);
        SDVariable ret = f().cumsum(reverse, 1);
        return Collections.singletonList(ret);
    }

}
