package ml.learner.neuralnet.initializers;

import java.util.Random;

import ml.learner.nerualnet.Layer;
import ml.utils.tracing.Trace;

public class DefaultWeightInitializer implements WeightInitializer {

	private Random _r = new Random();


	public DefaultWeightInitializer() {

	}


	@Override
	public void initializeWeights(Layer layer, double[][][] W) {
		Trace.log("DefaultWeightInitializer.initializeWeights() [begin]");

		if (layer.index() == 0) { return; }

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
				W[L][j][i] = _r.nextDouble() * 0.2;
			}
		}

		Trace.log("DefaultWeightInitializer.initializeWeights()", "Layer:", L);
		Trace.log("DefaultWeightInitializer.initializeWeights() [end]");

	}

}