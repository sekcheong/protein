package ml.learner.neuralnet;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import ml.data.Instance;
import ml.learner.neuralnet.functions.Function;
import ml.math.Vector;
import ml.utils.Format;
import ml.utils.tracing.Trace;


public class NeuralNet {

	// the network layers
	private Layer[] _layers = null;

	// the list holds the layers being added
	private List<Layer> _addLayers = new ArrayList<Layer>();

	// the network weight matrices
	private double[][][] _w;

	private double[][] _u;

	private double[][] _y;

	// the drop out mask
	private double[][] _m;

	private double[][] _delta;

	private double[][][][] _weightQueue;

	private Instance[] _tune;

	// the bias term
	private double _bias = 1;

	private int _epoch = 0;

	private boolean _dropOut;

	private Random _random = new Random();


	public NeuralNet() {}


	public double[][][] weights() {
		return _w;
	}


	public void weights(double[][][] weights) {
		_w = weights;
	}


	public double bias() {
		return _bias;
	}


	public void bias(double bias) {
		_bias = bias;
	}


	public void tune(Instance[] tuneSet) {
		_tune = tuneSet;
	}


	public Instance[] tune() {
		return _tune;
	}


	public int epoch() {
		return _epoch;
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
		_layers = _addLayers.toArray(new Layer[_addLayers.size()]);
		return layer;
	}


	public Layer[] layers() {
		return _layers;
	}


	public boolean dropOut() {
		return _dropOut;
	}


	public void dropOut(boolean dropOut) {
		_dropOut = dropOut;
	}


	public void train(Instance[] train, double eta, int maxEpoch, double epsilon) {
		train(train, null, eta, maxEpoch, epsilon, 0, 0);
	}


	public void train(Instance[] train, double eta, int maxEpoch, double epsilon, double alpha) {
		train(train, null, eta, maxEpoch, epsilon, alpha, 0);
	}


	public void train(Instance[] train, double eta, int maxEpoch, double epsilon, double alpha, double lambda) {
		train(train, null, eta, maxEpoch, epsilon, alpha, lambda);
	}


	public void train(Instance[] train, Instance[] tune, double eta, int maxEpoch, double epsilon, double alpha, double lambda) {
		// the current mean squared error
		double currMSE = 0;
		double prevMSE;

		// the current epoch
		_epoch = 0;

		Layer layers[] = this.layers();

		// _w[n][j][i] are weight matrices whose elements denote the value of
		// the synaptic weight that connects the j th neuron of layer (n) to the
		// i th neuron of layer (n - 1).
		_w = new double[layers.length][][];

		// _v[n][j] are vectors whose elements denote the weighted inputs
		// related to the j th neuron of layer n, and are defined by:
		_u = new double[layers.length][];

		// _y[n][j] are vectors whose elements denote the output of the j th
		// neuron related to the layer n. They are defined as:
		_y = new double[layers.length][];

		// The deltas for back propagations
		_delta = new double[_w.length][];

		// The dropout mask

		_m = new double[_w.length][];
		

		Instance[] trainCopy = new Instance[train.length];
		for (int i = 0; i < train.length; i++) {
			trainCopy[i] = train[i];
		}

		this.initialize(_w, _u, _m, _y, _delta);

		queueCurrentWeight(_w);

		Trace.log("W=[");
		Trace.log(Format.matrix(_w), "]");

		while (true) {

			shuffleArray(trainCopy);

			prevMSE = currMSE;

			for (Instance s : trainCopy) {

				feedForward(layers, _w, _m, _u, _y, s.features);
				backPropagation(layers, _w, _m, _u, _y, s.target, _delta, eta, alpha, lambda);

			}

			currMSE = computeMeanSquaredError(trainCopy);

			_epoch = _epoch + 1;

			Trace.log("Epoch: ", _epoch);
			
			if (_epoch >= maxEpoch) {
				Trace.log("Epoch topped out:", maxEpoch);
				break;
			}

			double d = Math.abs(prevMSE - currMSE);
			if (d <= epsilon) {
				Trace.log("epsilon = ", epsilon);
				Trace.log("Precision met: e = ", d);
				Trace.log("Epoch:", _epoch);
				break;
			}

		}

	}


	private void computeDropOutMask(double[][] m) {
		if (!this.dropOut()) return;
		for (int j = 0; j < m[0].length; j++) {
			m[0][j] = 1;
		}
		for (int l = 0; l < m.length; l++) {
			for (int j = 0; j < m[l].length; j++) {
				m[l][j] = flipCoin();
			}
		}
	}


	private int flipCoin() {
		if (_random.nextBoolean()) return 1;
		else return 0;
	}


	private void shuffleArray(Instance[] examples) {
		List<Instance> e = new ArrayList<Instance>();
		for (int i = 0; i < examples.length; i++) {
			e.add(examples[i]);
		}
		Collections.shuffle(e);
		for (int i = 0; i < e.size(); i++) {
			examples[i] = e.get(i);
		}
	}


	// allocates the necessary storage and initialize the weights for the learning session
	private void initialize(double[][][] w, double[][] u, double[][] m, double[][] y, double[][] delta) {
		Layer layers[] = this.layers();

		w[0] = new double[layers[0].units()][];
		m[0] = new double[layers[0].units()];

		// y[0] used as the input for the entire net and y[0][0] is the bias
		// term, this is little bit confusing but it will make the feedforward
		// easier
		y[0] = new double[w[0].length + 1];
		y[0][0] = _bias;

		// for each layer we initialize w[l], u[l], y[l], and delta[l]
		for (int l = 1; l < w.length; l++) {

			// create the neuron in layer l
			w[l] = new double[layers[l].units()][];
			m[l] = new double[layers[l].units()];

			// synaptic weight that connects the j th neuron of layer L to the
			// i th neuron of layer (L − 1) plus a bias term.
			for (int j = 0; j < w[l].length; j++) {
				w[l][j] = new double[layers[l - 1].units() + 1];
				m[l][j] = 1;
			}

			// u are vectors whose elements denote the weighted inputs related
			// to the j th neuron of layer L
			u[l] = new double[w[l].length];

			// y are vectors whose elements denote the output of the j th neuron
			// of layer L for hidden unit y also has a bias term at y[0];
			if (l < w.length - 1) {
				// for hidden layers add the bias term
				y[l] = new double[w[l].length + 1];
				y[l][0] = _bias;
			}
			else {
				y[l] = new double[w[l].length];
			}

			delta[l] = new double[w[l].length];

			layers[l].weightInitializer().initializeWeights(w[l]);
		}

	}


	private void feedForward(Layer layers[], double[][][] w, double[][] m, double[][] u, double[][] y, double[] x) {

		Function g = layers[1].activationFunction();
		for (int j = 0; j < y[0].length - 1; j++) {
			y[0][j + 1] = x[j];
		}
		y[0][0] = _bias;

		// compute the hidden units
		for (int l = 1; l < w.length; l++) {
			g = layers[l].activationFunction();
			for (int j = 0; j < w[l].length; j++) {
				double z = Vector.dot(w[l][j], y[l - 1]);
				u[l][j] = z;
				// if l is hidden layer offset by 1 for the bias term
				if (l < w.length - 1) {
					y[l][j + 1] = g.eval(z) * m[l][j];
				}
				else {
					y[l][j] = g.eval(z) * m[l][j];
				}
			}
			// if l is hidden layer set the bias
			if (l < w.length - 1) {
				y[l][0] = _bias;
			}
		}
	}


	private void backPropagation(Layer layers[], double[][][] w, double[][] m, double[][] u, double[][] y, double d[], double[][] delta, double eta, double alpha, double lambda) {
		// the output layer
		int l = w.length - 1;

		double[][][] w0 = lastWeight();

		Function g = layers[l].activationFunction();

		// calculate the deltas for the output layer
		for (int j = 0; j < w[l].length; j++) {
			delta[l][j] = (d[j] - y[l][j]) * g.diff(u[l][j]);
			// update weights
			for (int i = 0; i < w[l][j].length; i++) {
				double momentum = alpha * (w[l][j][i] - w0[l][j][i]);
				double weightDecay = eta * lambda * w[l][j][i];
				w[l][j][i] = w[l][j][i] + momentum + eta * delta[l][j] * y[l - 1][i] - weightDecay;
			}
		}

		// calculate the deltas for the hidden layers and the first layer
		for (l = w.length - 2; l >= 1; l--) {
			for (int j = 0; j < w[l].length; j++) {
				// compute delta[l]
				double z = 0;
				for (int k = 0; k < w[l + 1].length; k++) {
					z = z + delta[l + 1][k] * w[l + 1][k][j];
				}
				delta[l][j] = -z * g.diff(u[l][j]);
				// update weights
				for (int i = 0; i < w[l][j].length; i++) {
					double momentum = alpha * (w[l][j][i] - w0[l][j][i]);
					double weightDecay = eta * lambda * w[l][j][i];
					w[l][j][i] = w[l][j][i] + momentum + eta * delta[l][j] * y[l - 1][i] - weightDecay;
					// w[l][j][i] = w[l][j][i] * m[l-1][i];
				}
			}
		}

		queueCurrentWeight(w);
	}


	private double computeError(double[][][] w, double[] d, double[][] y) {
		double e = 0;
		int l = w.length - 1; // get the index of the output layout
		for (int j = 0; j < w[l].length; j++) {
			e = e + 0.5 * (d[j] - y[l][j]) * (d[j] - y[l][j]);
		}
		return e;
	}


	private double computeMeanSquaredError(Instance[] examples) {
		int p = 0;
		double[] se = new double[examples.length];
		for (Instance s : examples) {
			feedForward(_layers, _w, _m, _u, _y, s.features);
			se[p] = computeError(_w, s.target, _y);
			p = p + 1;
		}
		double mse = (1 / (double) se.length) * Vector.sigma(se);
		return mse;
	}


	private void queueCurrentWeight(double[][][] w) {
		if (_weightQueue == null) {
			_weightQueue = new double[2][][][];
		}
		if (_weightQueue[0] == null) {
			_weightQueue[0] = copyWeight(w);
		}
		else if (_weightQueue[1] == null) {
			_weightQueue[1] = copyWeight(w);
		}
		else {
			_weightQueue[0] = _weightQueue[1];
			_weightQueue[1] = copyWeight(w);
		}
	}


	private double[][][] lastWeight() {
		return _weightQueue[0];
	}


	private double[][][] copyWeight(double[][][] w) {
		double[][][] cpy = new double[w.length][][];
		for (int i = 0; i < w.length; i++) {
			if (w[i] == null)
				continue;
			cpy[i] = new double[w[i].length][];
			for (int j = 0; j < w[i].length; j++) {
				if (w[i][j] == null)
					continue;
				cpy[i][j] = new double[w[i][j].length];
				for (int k = 0; k < w[i][j].length; k++) {
					cpy[i][j][k] = w[i][j][k];
				}
			}
		}
		return cpy;
	}


	private static double[] threshold(double[] values) {
		double[] t = new double[values.length];
		int max = 0;
		for (int i = 0; i < values.length; i++) {
			if (values[i] > values[max]) {
				max = i;
			}
		}
		t[max] = 1;
		return t;
	}


	private double computeAccuracy(Instance[] tune) {
		int correct = 0;

		for (Instance t : tune) {
			double[] out = this.predict(t.features);

			out = threshold(out);

			boolean match = true;
			for (int i = 0; i < t.target.length; i++) {
				if (t.target[i] != out[i]) {
					match = false;
					break;
				}
			}

			if (match) {
				correct++;
			}

		}

		double acc = ((double) correct) / tune.length;
		return acc;
	}


	// private void filterOutput(double[] out) {
	// int max = 0;
	// for (int i = 1; i < out.length; i++) {
	// if (out[i] > out[max]) max = i;
	// out[i] = 0;
	// }
	// out[max] = 1.0;
	// }

	private double[] predict(double[] features, double[][][] w, double[][] m, double[][] u, double[][] y, double[] x) {
		feedForward(this.layers(), w, m, u, y, features);
		return y[y.length - 1];
	}


	public double[] predict(double[] features) {
		return predict(features, _w, _m, _u, _y, features);
	}

}
