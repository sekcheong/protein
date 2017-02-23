package ml.learner.nerualnet.config;

import ml.learner.nerualnet.Layer;

public interface ActivationFunction {
	public void layer(Layer layer);	
	public double[] compute(double[] potential);
	public double compute(double potential);
}