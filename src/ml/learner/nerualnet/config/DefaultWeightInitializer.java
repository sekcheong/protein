package ml.learner.nerualnet.config;

import ml.learner.nerualnet.Layer;
import ml.learner.nerualnet.Net;

public class DefaultWeightInitializer implements WeightInitializer {

	@Override
	public void initializeWeights(Layer layer) {
		Net net = layer.net();
		double[][][] w = net.W();
	}
	
}
