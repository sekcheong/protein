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
	private double _BIAS = -1.0;

	// the network layers
	private Layer[] _layers;

	// the list holds the layers being added
	private List<Layer> _addLayers = new ArrayList<Layer>();

	// the network weight matrices
	public double[][][] _W;


	public NeuralNet() {
	}


	public double[][][] W() {
		return _W;
	}


	public double bias() {
		return _BIAS;
	}


	public void bias(double bias) {
		_BIAS = bias;
	}


	public Layer addLayer(int units) {
		int index = _addLayers.size();
		Layer layer;

		if (index == 0) {
			layer = new Layer(units, units);
		}
		else {
			// input units is the number of units of the previous layer
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


	private void initialize(double[][][] w, double[][] v, double[][] y, double[][] delta) {
		Layer layers[] = this.layers();

		// initializes the weights for each hidden
		for (int l = 1; l < w.length; l++) {

			w[l] = new double[layers[l].units()][];
			v[l] = new double[w[l].length];
			delta[l] = new double[w[l].length + 1];

			if ((l != w.length - 1)) {
				// output for input and hidden units is previous layer's units plus the
				// bias unit
				y[l] = new double[w[l].length + 1];
			}
			else {
				// for output unit we don't need the bias unit
				y[l] = new double[w[l].length];
			}

			// number of units connecting from n - 1
			for (int j = 0; j < layers[l].units(); j++) {
				w[l][j] = new double[layers[l - 1].units()];
			}

			layers[l].weightInitializer().initializeWeights(w[l]);
		}

	}


	public void train(Instance[] examples, double eta, double precision, int maxEpoch) {
		// the current mean squared error
		double currMSE = 0;
		double prevMSE;

		// the current epoch
		int epoch = 0;

		Layer layers[] = this.layers();

		// W[n][j][i] are weight matrices whose elements denote the value of the
		// synaptic weight that connects the jth neuron of layer (n) to the ith
		// neuron of layer (n - 1).
		double[][][] W = new double[layers.length][][];

		// I[n][j] are vectors whose elements denote the weighted inputs related to
		// the jth neuron of layer n, and are defined by:
		double[][] I = new double[layers.length + 1][];

		// Y[n][j] are vectors whose elements denote the output of the jth neuron
		// related to the layer n. They are defined as:
		double[][] Y = new double[layers.length][];

		// X[i] is the input vector
		double[] X = new double[examples[0].features.length + 1];

		// E[k] is the error for kth example
		double[] E = new double[examples.length + 1];

		// The deltas for back propagations
		double[][] delta = new double[W.length][];

		// the scratch pad vector for computing the error for E[k]
		double[] e = new double[examples[0].target.length];

		this.initialize(W, I, Y, delta);
		this._W = W;

		// <debug>
		examples = debugNetValues(W);
		X = new double[examples[0].features.length + 1];
		E = new double[examples.length];
		e = new double[examples[0].target.length];
		// p = examples.length;
		// </debug>

		Trace.log("W=[");
		Trace.log(Format.matrix(W), "]");
		Trace.log("X=[", Format.matrix(X), "]");

		while (true) {

			prevMSE = currMSE;
			int p = 0;

			for (Instance example : examples) {

				setupInput(Y, example.features);
				feedForward(layers, W, I, Y);

				E[p] = computeError(example.target, Y, e);
				Trace.log("E[", p, "] = [", Format.matrix(E), "]");
				p = p + 1;
				backPropagation(layers, W, I, Y, example.target, eta, delta);

			}

			currMSE = (1 / (double) p) * Vector.sum(E);
			Trace.log("currMSE = ", currMSE);

			if (Math.abs(prevMSE - currMSE) <= precision) break;

			epoch = epoch + 1;
			if (epoch >= maxEpoch) break;

		}

	}


	private void setupInput(double[][] y, double x[]) {
		// Set Y[0] as the input vector and Y[0] as the bias term;
		y[0] = new double[x.length + 1];
		for (int i = 0; i < x.length; i++) {
			y[0][i + 1] = x[i];
		}
		y[0][0] = _BIAS;
	}


	private void feedForward(Layer layers[], double[][][] w, double[][] v, double[][] y) {

		for (int l = 1; l < w.length; l++) {
			Trace.log("w[", l, "] = [\n", Format.matrix(w[l]), "\n]");

			Function g = layers[l].activationFunction();

			for (int j = 0; j < w[l].length; j++) {

				v[l][j] = Vector.dot(w[l][j], y[l - 1]);

				// for all non output layers set the bias term
				if (l != w.length - 1) {
					y[l][0] = _BIAS;
					y[l][j + 1] = g.compute(v[l][j]);
				}
				else {
					// skip the bias for the output layer
					y[l][j] = g.compute(v[l][j]);
				}

			}

			Trace.log("I[", l, "] = [" + Format.matrix(v[l]), "]");
			Trace.log("Y[", l, "] = [" + Format.matrix(y[l]), "]");
		}

	}


	private void backPropagation(Layer layers[], double[][][] w, double[][] v, double[][] y, double d[], double eta, double[][] delta) {

		// the output layer
		int l = w.length - 1;

		Function g = layers[l].activationFunction();

		Trace.log("d = [", Format.matrix(d), "]");
		Trace.log("Y[", l, "] = ", "[", Format.matrix(y[l]), "]");

		// calculate the deltas for the output layer
		for (int j = 0; j < w[l].length; j++) {
			delta[l][j] = (d[j] - y[l][j]) * g.diff(v[l][j]);
		}

		// calculate the deltas for the hidden layers and the first layer
		for (l = w.length - 2; l >= 1; l--) {
			// delta[l] = new double[W[l].length + 1];
			for (int j = 0; j < w[l].length; j++) {
				double z = 0;
				for (int k = 1; k < w[l + 1].length; k++) {
					z = z + delta[l + 1][k] * w[l + 1][k][j];
				}
				delta[l][j] = -z * g.diff(v[l][j]);
			}
		}
	}


	private void adjustWeights(double[][][] w, double[][] y, int layer, double delta[][], double eta) {
		for (int l = w.length - 1; l >= 1; l--) {
			for (int j = 0; j < w[l].length; j++) {
				for (int i = 0; i < w[l][j].length; i++) {
					w[l][j][i] = w[l][j][i] + eta * delta[l][j] * y[l - 1][i];
				}
			}
		}
	}


	private double computeError(double[] d, double[][] y, double[] e) {
		int k = y.length - 1;
		Vector.sub(d, y[k], e);
		Vector.square(e, e);
		double error = 0.5 * Vector.sum(e);
		return error;
	}


	public double[] predict(double[] features) {
		return new double[0];
	}

}