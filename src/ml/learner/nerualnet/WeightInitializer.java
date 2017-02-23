package ml.learner.nerualnet;

import ml.learner.nerualnet.layers.AbstractLayer;

public interface WeightInitializer {	
	public void layer(AbstractLayer layer);
	public void initialize();
}