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
	private double[][][] _W;
	private double[][] _V;
	private double[][] _Y;
	private double[][] _DELTA;


	public NeuralNet() {}


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
			layer = new Layer(units, _addLayers.get(index - 1)
					.units());
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
		// W[1] = new double[3][3];
		// W[1][0] = new double[] { 0.2, 0.4, 0.5 };
		// W[1][1] = new double[] { 0.3, 0.6, 0.7 };
		// W[1][2] = new double[] { 0.4, 0.8, 0.3 };
		//
		// W[2] = new double[2][4];
		// W[2][0] = new double[] { -0.7, 0.6, 0.2, 0.7 };
		// W[2][1] = new double[] { -0.3, 0.7, 0.2, 0.8 };
		//
		// W[3] = new double[1][3];
		// W[3][0] = new double[] { 0.1, 0.8, 0.5 };
		//
		// Instance[] ex = new Instance[1];
		// ex[0] = new Instance();
		//
		// ex[0].features = new double[] { 0.3, 0.7 };
		// ex[0].target = new double[] { 0.9 };

		Instance[] ex = new Instance[4];

		ex[0] = new Instance();
		ex[0].features = new double[] { 0.2, 0.9, 0.4 };
		ex[0].target = new double[] { 0.7, 0.3 };

		ex[1] = new Instance();
		ex[1].features = new double[] { 0.1, 0.3, 0.5 };
		ex[1].target = new double[] { 0.6, 0.4 };

		ex[2] = new Instance();
		ex[2].features = new double[] { 0.9, 0.7, 0.8 };
		ex[2].target = new double[] { 0.9, 0.5 };

		ex[3] = new Instance();
		ex[3].features = new double[] { 0.6, 0.4, 0.3 };
		ex[3].target = new double[] { 0.2, 0.8 };

		return ex;
	}


	public void train(Instance[] examples, double eta, double alpha, double lambda, double epsilon, int maxEpoch) {
		// the current mean squared error
		double currMSE = 0;
		double prevMSE;

		// the current epoch
		int epoch = 0;

		Layer layers[] = this.layers();

		// W[n][j][i] are weight matrices whose elements denote the value of the
		// synaptic weight that connects the jth neuron of layer (n) to the
		// ith neuron of layer (n - 1).
		_W = new double[layers.length][][];

		// V[n][j] are vectors whose elements denote the weighted inputs related
		// to
		// the jth neuron of layer n, and are defined by:
		_V = new double[layers.length][];

		// Y[n][j] are vectors whose elements denote the output of the jth
		// neuron related to the layer n. They are defined as:
		_Y = new double[layers.length][];

		// The deltas for back propagations
		_DELTA = new double[_W.length][];

		// E[k] is the error for kth example
		double[] E = new double[examples.length + 1];

		// the scratch pad vector for computing the error for E[k]
		double[] e = new double[examples[0].target.length];

		this.initialize(_W, _V, _Y, _DELTA);

		// // <debug>
		// examples = debugNetValues(W);
		// E = new double[examples.length];
		// e = new double[examples[0].target.length];
		// // </debug>

		Trace.log("W=[");
		Trace.log(Format.matrix(_W), "]");

		while (true) {

			prevMSE = currMSE;
			int p = 0;

			for (Instance s : examples) {

				feedForward(layers, _W, _V, _Y, s.features);

				E[p] = computeError(_W, s.target, _Y, e);
				// Trace.log("E[", p, "] = [", Format.matrix(E), "]");
				p = p + 1;

				backPropagation(layers, _W, _V, _Y, s.target, _DELTA, eta, alpha, lambda);

			}

			currMSE = (1 / (double) p) * Vector.sigma(E);
			// Trace.log("currMSE = ", currMSE);
			double d = Math.abs(prevMSE - currMSE);
			if (d <= epsilon) {
				Trace.log("Precision met: e = ", d);
				// break;
			}

			// Trace.log("epsilon = ", prevMSE - currMSE);
			epoch = epoch + 1;
			if (epoch >= maxEpoch) {
				Trace.log("Epoch topped out:", maxEpoch);
				break;
			}

		}

	}
	
	
	private void initialize(double[][][] w, double[][] v, double[][] y, double[][] delta) {
		Layer layers[] = this.layers();

		w[0] = new double[layers[0].units() + 1][];
		y[0] = new double[layers[0].units() + 1];

		for (int l = 1; l < w.length; l++) {

			w[l] = new double[layers[l].units() + 1][];

			for (int j = 0; j < w[l].length; j++) {
				w[l][j] = new double[w[l - 1].length];
			}

			v[l] = new double[w[l - 1].length];
			y[l] = new double[w[l].length];
			delta[l] = new double[w[l].length];

			layers[l].weightInitializer()
					.initializeWeights(w[l]);
		}

	}


	private void feedForward(Layer layers[], double[][][] w, double[][] v, double[][] y, double[] x) {

		Function g = layers[1].activationFunction();
		for (int j = 0; j < y[0].length - 1; j++) {
			y[0][j + 1] = x[j];
		}
		y[0][0] = _BIAS;

		// compute the hidden units
		for (int l = 1; l < w.length; l++) {
			g = layers[l].activationFunction();
			for (int j = 0; j < w[l].length; j++) {
				double z = Vector.dot(w[l][j], y[l - 1]);
				v[l][j] = z;
				y[l][j] = g.compute(z);
			}
			y[l][0] = _BIAS;
			v[l][0] = _BIAS;
		}

	}


	private void backPropagation(Layer layers[], double[][][] w, double[][] v, double[][] y, double d[], double[][] delta, double eta, double alpha, double lambda) {
		// the output layer
		int n = w.length - 1;
		Function g = layers[n].activationFunction();

		// calculate the deltas for the output layer
		for (int j = 0; j < w[n].length; j++) {
			delta[n][j] = (d[j] - y[n][j]) * g.diff(v[n][j]);
		}

		// calculate the deltas for the hidden layers and the first layer
		for (n = w.length - 2; n >= 1; n--) {
			for (int j = 0; j < w[n].length; j++) {
				double z = 0;
				for (int k = 1; k < w[n + 1].length; k++) {
					z = z + delta[n + 1][k] * w[n + 1][k][j];
				}
				delta[n][j] = -z * g.diff(v[n][j]);
			}
		}

		updateWeights(layers, w, y, delta, eta, lambda, alpha);

	}


	private void copyWeights(double[][][] dest, double[][][] src) {

	}


	private void updateWeights(Layer layers[], double[][][] w, double[][] y, double[][] delta, double eta, double alpha, double lambda) {
		for (int n = w.length - 1; n >= 1; n--) {
			for (int j = 0; j < w[n].length; j++) {
				for (int i = 1; i < w[n][j].length; i++) {
					w[n][j][i] = w[n][j][i] + eta * delta[n][j] * y[n - 1][j];
				}
			}
		}
	}


	private double computeError(double[][][] w, double[] d, double[][] y, double[] e) {
		// use the y value of the output
		int k = y.length - 1;
		Vector.sub(d, y[k], e);
		Vector.square(e, e);
		double error = 0.5 * Vector.sigma(e);
		return error;
	}


	public double[] predict(double[] features) {
		Layer[] layers = _layers;

		for (int n = 1; n < _W.length; n++) {

			Function g = layers[n].activationFunction();

			for (int j = 0; j < _W[n].length; j++) {

				_V[n][j] = Vector.dot(_W[n][j], _Y[n - 1]);

				// for all non output layers set the bias term
				if (n != _W.length - 1) {
					_Y[n][0] = _BIAS;
					_Y[n][j + 1] = g.compute(_V[n][j]);
				}
				else {
					// skip the bias for the output layer
					_Y[n][j] = g.compute(_V[n][j]);
				}

			}
		}
		return _Y[_W.length - 1];
	}

}

// Α α Alpha a
// Β β Beta b
// Γ γ Gamma g
// Δ δ Delta d
// Ε ε Epsilon e
// Ζ ζ Zeta z
// Η η Eta h
// Θ θ Theta th
// Ι ι Iota i
// Κ κ Kappa k
// Λ λ Lambda l
// Μ μ Mu m
// Ν ν Nu n
// Ξ ξ Xi x
// Ο ο Omicron o
// Π π Pi p
// Ρ ρ Rho r
// Σ σ,ς * Sigma s
// Τ τ Tau t
// Υ υ Upsilon u
// Φ φ Phi ph
// Χ χ Chi ch
// Ψ ψ Psi ps
// Ω ω Omega o