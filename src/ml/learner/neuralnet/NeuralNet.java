package ml.learner.neuralnet;

import java.util.ArrayList;
import java.util.List;
import ml.data.Instance;
import ml.learner.neuralnet.functions.Function;
import ml.math.Vector;
import ml.utils.Format;
import ml.utils.tracing.Trace;

public class NeuralNet {

	// the bias term
	private static double BIAS = -1.0;

	// the network layers
	private Layer[] _layers;

	// max number of epochs
	private int _maxEpoch = 5;

	// list holds the layers being added
	private List<Layer> _addLayers = new ArrayList<Layer>();

	public double[][][] _W;


	public NeuralNet() {
	}


	public double[][][] W() {
		return _W;
	}


	public Layer addLayer(int units) {
		int index = _addLayers.size();
		Layer layer;

		if (index == 0) {
			layer = new Layer(units, units);
		}
		else {
			// number of intput units is the number of unites of the previous
			// layer
			layer = new Layer(units, _addLayers.get(index - 1).units());
		}

		layer.index(_addLayers.size());
		layer.net(this);
		_addLayers.add(layer);

		return layer;
	}


	public Layer[] layers() {
		if (_layers == null) {
			_layers = _addLayers.toArray(new Layer[_addLayers.size()]);
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


	private Instance[] debugNetValues(double[][][] W) {
		W[1] = new double[3][3];
		W[1][0] = new double[] { 0.2, 0.4, 0.5 };
		W[1][1] = new double[] { 0.3, 0.6, 0.7 };
		W[1][2] = new double[] { 0.4, 0.8, 0.3 };

		W[2] = new double[2][4];
		W[2][0] = new double[] { -0.7, 0.6, 0.2, 0.7 };
		W[2][1] = new double[] { -0.3, 0.7, 0.2, 0.8 };

		W[3] = new double[1][3];
		W[3][0] = new double[] { 0.1, 0.8, 0.5 };

		Instance[] ex = new Instance[1];
		ex[0] = new Instance();

		ex[0].features = new double[] { 0.3, 0.7 };
		ex[0].target = new double[] { 0.9 };

		// Instance[] ex = new Instance[4];
		//
		// ex[0] = new Instance();
		// ex[0].features = new double[] { 0.2, 0.9, 0.4 };
		// ex[0].target = new double[] { 0.7, 0.3 };
		//
		// ex[1] = new Instance();
		// ex[1].features = new double[] { 0.1, 0.3, 0.5 };
		// ex[1].target = new double[] { 0.6, 0.4 };
		//
		// ex[2] = new Instance();
		// ex[2].features = new double[] { 0.9, 0.7, 0.8 };
		// ex[2].target = new double[] { 0.9, 0.5 };
		//
		// ex[3] = new Instance();
		// ex[3].features = new double[] { 0.6, 0.4, 0.3 };
		// ex[3].target = new double[] { 0.2, 0.8 };

		return ex;
	}


	private void initialize(double[][][] W, double[][] I, double[][] Y) {
		Layer layers[] = this.layers();

		// initializes the weights for each hidden
		for (int n = 1; n < W.length; n++) {

			W[n] = new double[layers[n].units()][];
			I[n] = new double[W[n].length];

			if ((n < W.length - 1)) {
				// output for input and hidden unit is previous layer's units plus a
				// bias unit
				Y[n] = new double[W[n].length + 1];
			}
			else {
				// for output unit we don't need the bias unit
				Y[n] = new double[W[n].length];
			}

			// number of units connecting from n - 1
			for (int j = 0; j < layers[n].units(); j++) {
				W[n][j] = new double[layers[n - 1].units()];
			}

			layers[n].weightInitializer().initializeWeights(W[n]);
		}

	}


	public void train(Instance[] examples, double eta, double precision, int maxEpoch) {
		// the current mean squared error
		double currMSE = 0;
		double prevMSE;

		// the current epoch
		int epoch = 0;

		Layer layers[] = this.layers();

		// W[n][j][i] are weight matrices whose elements denote the value of
		// the synaptic weight that connects the jth neuron of layer (n) to
		// the ith neuron of layer (n - 1).
		double[][][] W = new double[layers.length][][];

		// I[n][j] are vectors whose elements denote the weighted inputs
		// related to the jth neuron of layer n, and are defined by:
		double[][] I = new double[layers.length + 1][];

		// Y[n][j] are vectors whose elements denote the output of the jth
		// neuron related to the layer n. They are defined as:
		double[][] Y = new double[layers.length][];

		// X[i] is the input vector
		double[] X = new double[examples[0].features.length + 1];

		// E[k] is the error for kth example
		double[] E = new double[examples.length + 1];

		// the scratch pad vector for computing the error for E[k]
		double[] e = new double[examples[0].target.length];

		this.initialize(W, I, Y);
		this._W = W;

		// <debug>
		examples = debugNetValues(W);
		X = new double[examples[0].features.length + 1];
		E = new double[examples.length];
		e = new double[examples[0].target.length];
		// p = examples.length;
		// </debug>

		Trace.log("W=[");
		Trace.log(Format.matrix(W));
		Trace.log("]");
		Trace.log("X=[", Format.matrix(X), "]");

		while (true) {

			prevMSE = currMSE;
			int p = 0;

			for (Instance example : examples) {

				feedForward(layers, W, I, Y, example.features);

				E[p] = computeError(example.target, Y, e);
				p = p + 1;

				Trace.log("E[", p, "]=[", Format.matrix(E), "]");
				backPropagation(layers, W, I, Y, X, example.target, eta);

			}

			currMSE = (1 / (double) p) * Vector.sum(E);
			Trace.log("currMSE = ", currMSE);

			if (Math.abs(prevMSE - currMSE) <= precision) break;

			epoch = epoch + 1;
			if (epoch >= maxEpoch) break;

		}

	}


	private void feedForward(Layer layers[], double[][][] W, double[][] I, double[][] Y, double X[]) {
		// Y[n][1..J] = X
		Y[0] = new double[X.length + 1];
		for (int i = 0; i < X.length; i++) {
			Y[0][i + 1] = X[i];
		}
		Y[0][0] = BIAS;

		for (int n = 1; n < W.length; n++) {
			Trace.log("W[", n, "] = [\n", Format.matrix(W[n]), "\n]");

			Function g = layers[n].activationFunction();

			for (int j = 0; j < W[n].length; j++) {

				I[n][j] = Vector.dot(W[n][j], Y[n - 1]);

				// for all non output layers set the bias term
				if (n != W.length - 1) {
					Y[n][0] = BIAS;
					Y[n][j + 1] = g.compute(I[n][j]);
				}
				else {
					// skip the bias for the output layer
					Y[n][j] = g.compute(I[n][j]);
				}

			}

			Trace.log("I[", n, "] = [" + Format.matrix(I[n]), "]");
			Trace.log("Y[", n, "] = [" + Format.matrix(Y[n]), "]");
		}

	}


	private void backPropagation(Layer layers[], double[][][] W, double[][] I, double[][] Y, double X[], double d[], double eta) {

		double[][] delta = new double[W.length][];

		// the output layer
		int l = W.length - 1;
		delta[l] = new double[W[l].length + 1];
		Function g = layers[l].activationFunction();

		Trace.log("d=", Format.matrix(d));
		Trace.log("Y[", l, "]=", Format.matrix(Y[l]));

		for (int j = 0; j < W[l].length; j++) {

		}

		for (int j = 0; j < W[l].length; j++) {
			delta[l][j] = (d[j] - Y[l][j]) * g.diff(I[l][j]);
		}

		Trace.log("delta[", l, "]=", Format.matrix(delta[l]));

		for (int n = W.length - 2; n > 0; n--) {

			Trace.log("backprop: layer = ", n);

			delta[n] = new double[W[n].length];
			for (int j = 0; j < delta[n].length; j++) {
				for (int k = 0; k < delta[n + 1].length; k++) {
					delta[n][j] += -(delta[n + 1][k] * W[n + 1][k][j]);
				}
				delta[n][j] = delta[n][j] * g.diff(I[n + 1][j]);
			}
		}
	}


	private void computeDelta(double[][][] W, double[][] Y, double[][] X, double d, double[][] delta) {
		// for (int n = 0; n < W.length; n++) {
		// for (int j = 0; j < W[n].length; j++) {
		// for (int i = 0; i < W[n][j].length; j++) {
		// // W[n][j][i] = W[n][j][i] + eta * delta[n][j]*X[i];
		// }
		// }
		// }
	}


	private double computeError(double[] d, double[][] y, double[] e) {
		Vector.sub(d, y[y.length - 1], e);
		Vector.square(e, e);
		double error = 0.5 * Vector.sum(e);
		return error;
	}


	public double[] predict(double[] features) {
		return new double[0];
	}

}