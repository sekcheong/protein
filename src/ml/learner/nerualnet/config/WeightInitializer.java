package ml.learner.nerualnet.config;

import ml.learner.nerualnet.Layer;

public interface WeightInitializer {		
	public void initializeWeights(Layer layer);
}