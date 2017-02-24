package ml.learner.nerualnet;

import java.util.ArrayList;
import java.util.List;
import ml.data.Instance;
import ml.learner.nerualnet.functions.Function;
import ml.math.Vector;

public class Net {

	// the bias term
	private static double BIAS = -1.0;

	// the network layers
	private Layer[] _layers;

	// max number of epochs
	private int _maxEpoch = 5;

	// list holds the layers being added
	private List<Layer> _tempLayers = new ArrayList<Layer>();

	public Net() {}

	public void addLayer(Layer layer) {
		layer.index(_tempLayers.size());
		_tempLayers.add(layer);
		layer.net(this);
	}

	public Layer[] layers() {
		if (_layers == null) {
			_layers = _tempLayers.toArray(new Layer[_tempLayers.size()]);
		}
		return _layers;
	}

	private void setBias(double[][][] w) {
		for (int n = 1; n < w.length; n++) {
			for (int j = 0; j < w[n].length; j++) {
				w[n][j][0] = BIAS;
			}
		}
	}

	private void debugWeights(double[][][] W) {
		W[1] = new double[3][3];

		W[2] = new double[2][4];

		W[3] = new double[1][3];
	}

	public void train(Instance[] examples, double eta, double precision, int maxEpoch) {
		// the current mean squared error
		double currMSE = 0;
		double prevMSE;

		// the current epoch
		int epoch = 0;

		// p = number of examples
		int p = examples.length;
		Layer layers[] = this.layers();

		// W[n][j][i] are weight matrices whose elements denote the value of
		// the synaptic weight that connects the jth neuron of layer (n) to
		// the ith neuron of layer (n - 1).
		double[][][] W = new double[layers.length][][];

		// I[n][j] are vectors whose elements denote the weighted inputs
		// related to the jth neuron of layer n, and are defined by:
		double[][] I = new double[layers.length][];

		// Y[n][j] are vectors whose elements denote the output of the jth
		// neuron related to the layer n. They are defined as:
		double[][] Y = new double[layers.length][];

		// X[i] is the input vector
		double[] X = new double[examples[0].features.length];

		// E[k] is the error for kth example
		double[] E = new double[examples.length];

		// the scratch pad vector for computing the error for E[k]
		double[] e = new double[examples[0].target.length];

		this.initialize(W, I, Y);

		while (true) {

			prevMSE = currMSE;
			currMSE = 0;

			for (int k = 0; k < p; k++) {

				Instance example = examples[k];
				makeInputVector(X, example.features);

				computeForward(layers, W, I, Y, X);

				E[k] = computeError(example.target, Y[Y.length - 1], e);

				computeBackward(layers, W, I, Y, X);

			}

			currMSE = (1 / p) * Vector.sum(E);

			if (Math.abs(prevMSE - currMSE) <= precision) break;

			epoch = epoch + 1;
			if (epoch >= maxEpoch) break;

		}

	}

	private void initialize(double[][][] W, double[][] I, double[][] Y) {
		Layer layers[] = this.layers();

		// initializes the weights for each hidden
		for (int n = 1; n < W.length; n++) {

			W[n] = new double[layers[n].units() + 1][];
			Y[n] = new double[layers[n - 1].units() + 1];
			I[n] = new double[layers[n - 1].units() + 1];

			// number of units connecting from n - 1
			for (int j = 0; j < layers[n].units() + 1; j++) {
				W[n][j] = new double[layers[n - 1].units() + 1];
			}

			layers[n].initWeight(W);
		}

		setBias(W);

		for (int n = 1; n < layers.length; n++) {
			layers[n].printWeightMatrix(W[n]);
		}
	}

	private void computeForward(Layer layers[], double[][][] W, double[][] I, double[][] Y, double X[]) {
		double[] x;
		double z;

		// input layer
		Function g = layers[0].activationFunction();
		for (int j = 0; j < W[1].length; j++) {
			z = 0;
			for (int i = 0; i < W[1][j].length; i++) {
				z = z + W[1][j][i] * X[i];
			}
			I[1][j] = z;
		}
		
		g.compute(I[1], Y[1]);
		I[1][0] = BIAS;
		

		// hidden layers
		for (int n = 2; n < W.length; n++) {
			for (int j = 0; j < W[n].length; j++) {
				z = 0;
				for (int i = 0; i < W[n][j].length; i++) {
					z = z + W[n][j][i] * Y[n - 1][i];
				}
				I[n][j] = z;
			}
			I[n][0] = BIAS;
			g = layers[n].activationFunction();
			g.compute(I[n], Y[n]);
		}

	}

	private double computeError(double[] d, double[] y, double[] e) {
		Vector.sub(d, y, e);
		Vector.square(e, e);
		double error = 0.5 * Vector.sum(e);
		return error;
	}

	private void computeBackward(Layer layers[], double[][][] W, double[][] I, double[][] Y, double X[]) {

	}

	private static void makeInputVector(double[] x, double[] features) {
		x[0] = BIAS;
		for (int i = 0; i < features.length; i++) {
			x[i + 1] = features[i];
		}
	}

	public double[] predict(double[] features) {
		return new double[0];
	}

}