package ml.learner.neuralnet;

import java.util.ArrayList;
import java.util.List;
import ml.data.Instance;
import ml.learner.neuralnet.functions.Function;
import ml.learner.neuralnet.functions.Sigmoid;
import ml.learner.neuralnet.initializers.DefaultWeightInitializer;
import ml.learner.neuralnet.initializers.WeightInitializer;
import ml.math.Vector;
import ml.utils.Format;
import ml.utils.tracing.Trace;

public class NeuralNet {

	// the network layers
	private Layer[] _layers;

	// the list holds the layers being added
	private List<Layer> _addLayers = new ArrayList<Layer>();

	// the network weight matrices
	public double[][][] _w;

	private double[][] _u;

	private double[][] _y;

	private double[][] _delta;

	// the bias term
	private double _bias = 1;

//	// the default activation function
//	Function _sigmoid = new Sigmoid();
//
//	// the default weight initializer
//	WeightInitializer _weightInit = new DefaultWeightInitializer();


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


	private void debugNetValues(double[][][] W) {
		W[1] = new double[3][3];
		W[1][0] = new double[] { 0.2, 0.4, 0.5 };
		W[1][1] = new double[] { 0.3, 0.6, 0.7 };
		W[1][2] = new double[] { 0.4, 0.8, 0.3 };

		W[2] = new double[2][4];
		W[2][0] = new double[] { -0.7, 0.6, 0.2, 0.7 };
		W[2][1] = new double[] { -0.3, 0.7, 0.2, 0.8 };

		W[3] = new double[1][3];
		W[3][0] = new double[] { 0.1, 0.8, 0.5 };

		// Instance[] ex = new Instance[1];
		// ex[0] = new Instance();
		//
		// ex[0].features = new double[] { 0.3, 0.7 };
		// ex[0].target = new double[] { 0.9 };

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
	}


	public void train(Instance[] examples, double eta, double alpha, double lambda, double epsilon, int maxEpoch) {
		// the current mean squared error
		double currMSE = 0;
		double prevMSE;

		// the current epoch
		int epoch = 0;

		Layer layers[] = this.layers();

		// _w[n][j][i] are weight matrices whose elements denote the value of the
		// synaptic weight that connects the jth neuron of layer (n) to the
		// ith neuron of layer (n - 1).
		_w = new double[layers.length][][];

		// _v[n][j] are vectors whose elements denote the weighted inputs related
		// to the jth neuron of layer n, and are defined by:
		_u = new double[layers.length][];

		// _y[n][j] are vectors whose elements denote the output of the jth
		// neuron related to the layer n. They are defined as:
		_y = new double[layers.length][];

		// The deltas for back propagations
		_delta = new double[_w.length][];

		// se[k] is the error for kth sample
		double[] se = new double[examples.length];
		
		
		this.initialize(_w, _u, _y, _delta);

//		_w = new double[3][][];
//		_w[0] = new double[2][2];
//		_w[1] = new double[2][3];
//		_w[2] = new double[2][3];
//
//		_w[1][0][0] = .35;
//		_w[1][0][1] = .15;
//		_w[1][0][2] = .20;
//
//		_w[1][1][0] = .35;
//		_w[1][1][1] = .25;
//		_w[1][1][2] = .30;
//
//		_w[2][0][0] = .60;
//		_w[2][0][1] = .40;
//		_w[2][0][2] = .45;
//
//		_w[2][1][0] = .60;
//		_w[2][1][1] = .50;
//		_w[2][1][2] = .55;

		Trace.log("W=[");
		Trace.log(Format.matrix(_w), "]");


		while (true) {
			
			Trace.log("");
			Trace.log("Epoch: ", epoch);

			prevMSE = currMSE;
			
			// total number of samples in the batch
			int p = 0;
			
			for (Instance s : examples) {

				feedForward(layers, _w, _u, _y, s.features);
				backPropagation(layers, _w, _u, _y, s.target, _delta, eta, alpha, lambda);

				// recalculates the predicted y based on the updated weights
				feedForward(layers, _w, _u, _y, s.features);
				
				// computes the deviation of prediction from target
				double e = computeError(_w, s.target, _y);
				se[p] = e;
				p = p + 1;
				
				Trace.log("E[", p - 1, "] = ", se[p - 1]);
			}

			// mean squre error for the entire batch of samples 
			currMSE = (1 / (double) p) * Vector.sigma(se);
			
			Trace.log("currMSE = ", currMSE);
			Trace.log("w = ", Format.matrix(_w));
			
			double d = Math.abs(prevMSE - currMSE);
			if (d <= epsilon) {
				Trace.log("Precision met: e = ", d);
				//break;
			}

			// Trace.log("epsilon = ", prevMSE - currMSE);
			epoch = epoch + 1;
			if (epoch >= maxEpoch) {
				Trace.log("Epoch topped out:", maxEpoch);
				break;
			}

		}

	}


	// allocates the nessary storage and initialize the weights for the learning
	// session
	private void initialize(double[][][] w, double[][] u, double[][] y, double[][] delta) {
		Layer layers[] = this.layers();

		w[0] = new double[layers[0].units()][];

		// y[0] used as the input for the entire net and y[0][0] is the bias
		// term
		// this is little bit confusing but it will make the feedforward easire
		y[0] = new double[w[0].length + 1];
		y[0][0] = _bias;

		// for each layer we initialize w[l], u[l], y[l], and delta[l]
		for (int l = 1; l < w.length; l++) {

			// create the neuron in layer l
			w[l] = new double[layers[l].units()][];

			// synaptic weight that connects the jth neuron of layer L to the
			// ith neuron of layer (L − 1) plus a bias term.
			for (int j = 0; j < w[l].length; j++) {
				w[l][j] = new double[layers[l - 1].units() + 1];
			}

			// u are vectors whose elements denote the weighted inputs related
			// to the jth neuron of layer L
			u[l] = new double[w[l].length];

			// y are vectors whose elements denote the output of thejth neuron
			// of layer L
			// for hidden unit y also has a bias term at y[0];
			if (l < w.length - 1) {
				// for hidden layerso add the bias term
				y[l] = new double[w[l].length + 1];
				y[l][0] = _bias;
			}
			else {
				y[l] = new double[w[l].length];
			}

			delta[l] = new double[w[l].length];

			layers[l].weightInitializer()
					.initializeWeights(w[l]);
		}

	}


	private void feedForward(Layer layers[], double[][][] w, double[][] u, double[][] y, double[] x) {

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
				// if l is hidden layer off set by 1 for the bias term
				if (l < w.length - 1) {
					y[l][j + 1] = g.eval(z);
				}
				else {
					y[l][j] = g.eval(z);
				}
			}
			// if l is hidden layer set the bias
			if (l < w.length - 1) {
				y[l][0] = _bias;
			}
		}
	}


	private void backPropagation(Layer layers[], double[][][] w, double[][] u, double[][] y, double d[], double[][] delta, double eta, double alpha, double lambda) {
		// the output layer
		int l = w.length - 1;
		Function g = layers[l].activationFunction();

		// calculate the deltas for the output layer
		for (int j = 0; j < w[l].length; j++) {
			
			delta[l][j] = (d[j] - y[l][j]) * g.diff(u[l][j]);
			
			// update weights
			for (int i = 0; i < w[l][j].length; i++) {
				w[l][j][i] = w[l][j][i] + eta * delta[l][j] * y[l - 1][i];
			}
			
		}

		// calculate the deltas for the hidden layers and the first layer
		for (l = w.length - 2; l >= 1; l--) {
			
			for (int j = 0; j < w[l].length; j++) {
				
				//compute delta[l] 
				double z = 0;
				for (int k = 0; k < w[l + 1].length; k++) {
					z = z + delta[l + 1][k] * w[l + 1][k][j];
				}
				delta[l][j] = -z * g.diff(u[l][j]);
				
				// update weights
				for (int i = 0; i < w[l][j].length; i++) {
					w[l][j][i] = w[l][j][i] + eta * delta[l][j] * y[l - 1][i];
				}
			
			}
			
		}
	}


	private double computeError(double[][][] w, double[] d, double[][] y) {
		double e = 0;
		int l = w.length - 1;
		for (int j = 0; j < w[l].length; j++) {
			e = e + 0.5 * (d[j] - y[l][j]) * (d[j] - y[l][j]);
		}
		return e;
	}


	public double[] predict(double[] features) {
		Layer[] layers = _layers;

		for (int n = 1; n < _w.length; n++) {

			Function g = layers[n].activationFunction();

			for (int j = 0; j < _w[n].length; j++) {

				_u[n][j] = Vector.dot(_w[n][j], _y[n - 1]);

				// for all non output layers set the bias term
				if (n != _w.length - 1) {
					_y[n][0] = _bias;
					_y[n][j + 1] = g.eval(_u[n][j]);
				}
				else {
					// skip the bias for the output layer
					_y[n][j] = g.eval(_u[n][j]);
				}

			}
		}
		return _y[_w.length - 1];
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