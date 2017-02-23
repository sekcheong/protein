package ml.learner.nerualnet;

import ml.learner.nerualnet.layers.AbstractLayer;

public interface WeightInitializer {	
	public void initializeWeights(AbstractLayer layer);
}