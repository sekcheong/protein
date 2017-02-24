package ml.learner.neuralnet.initializers;

import ml.learner.nerualnet.Layer;

public interface WeightInitializer {		
	public void initializeWeights(Layer layer, double[][][] W);
}