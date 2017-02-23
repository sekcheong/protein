package ml.learner.nerualnet.config;

import ml.learner.nerualnet.Layer;
import ml.learner.nerualnet.Net;

public class DefaultWeightInitializer implements WeightInitializer {

	public DefaultWeightInitializer() {
	}

	@Override
	public void initializeWeights(Layer layer) {

		if (layer.index() == 0) { return; }

		double[][][] W = layer.net().W();

		// get layer the initializer belongs to
		int L = layer.index();

		// the number of unites in layer L
		int J = layer.units();

		// the number of units in layer L-1
		int I = layer.net().layers()[L - 1].units();

		// initialize the weights of the neuron
		// connecting from L-1 layer to L
		for (int j = 0; j < J; j++) {
			for (int i = 0; i < I; i++) {
				W[L][j][i] = Math.random() * 0.3;
			}
		}
	}

}