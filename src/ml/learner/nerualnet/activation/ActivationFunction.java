package ml.learner.nerualnet.activation;

import ml.learner.nerualnet.layers.AbstractLayer;

public interface ActivationFunction {
	public void layer(AbstractLayer layer);	
	public double[] compute(double[] potential);
	public double compute(double potential);
}