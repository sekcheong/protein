package ml.learner.nerualnet;

import java.util.ArrayList;
import java.util.List;
import ml.data.Instance;


public class Net {

	// the bias term
	private static double BIAS = -1.0;

	// W[L,j,i] are weight matrices where L is the layer number, j is the
	// j_th neuron of L, and i is the i_th neuron of layer L-1
	public double[][][] _W = null;

	// I[L,j] are vectors whose element denote weighted input of j_th
	// neuron of layer L

	// Y[L,j] are vectors whose elements denote the output of the j_th
	// neuron of layer L

	// the error
	private double[] E = null;

	// the network layers
	private Layer[] _layers;

	// max number of epochs
	private int _maxEpoch = 5;

	// the target precision
	private double _epsilon = 0.4;

	// the learning rate
	private double _eta = 0.05;

	// list holds the layers being added
	private List<Layer> _tempLayers = new ArrayList<Layer>();

	
	public Net() {
	}

	
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

	
	public void train(Instance[] examples, double eta, double precision, int maxEpoch) {
		// the current mean squared error
		double currMSE = 0;
		double prevMSE;
		int p = examples.length;
		int epoch = 0;

		Layer layers[] = this.layers();
		double[][][] W = new double[layers.length][][];
		double[] X = new double[examples[0].features.length + 1];
		double[][] Y = new double[layers.length][];
		double[][] I = new double[layers.length][];

		this.initialize(W, X, Y, I);

		while (true) {

			prevMSE = currMSE;
			currMSE = 0;

			for (int k = 0; k < p; k++) {

				Instance example = examples[k];
				makeInputVector(X, example.features);

				forwardComputation(W, X);
				backwardComputation(W, I, X);
				
				currMSE = currMSE + computeError(example.target, Y[Y.length - 1]);

			}

			currMSE = currMSE / p;
			if (Math.abs(prevMSE - currMSE) <= precision) break;

			epoch = epoch + 1;
			if (epoch >= maxEpoch) break;

		}

	}

	
	private double computeError(double[] d, double[] y) {
		
		return 0;
	}

	
	private void initialize(double[][][] W, double[] X, double[][] Y, double[][] I) {
		Layer layers[] = this.layers();

		// number of units for the output layer
		int outputs = layers[layers.length - 1].units();

		// initializes the weights for each hidden
		for (int n = 1; n < W.length; n++) {
			
			// number of units in layer n
			int units = layers[n].units();
			W[n] = new double[units][];
			Y[n] = new double[units];
			I[n] = new double[units];
			
			// number of units connecting from n - 1
			for (int j = 0; j < units; j++) {
				W[n][j] = new double[layers[n - 1].units()];
				
			}

			layers[n].initWeight(W);
		}

		setBias(W);

		for (int n = 1; n < layers.length; n++) {
			layers[n].printWeightMatrix(W[n]);
		}
	}

	
	private void forwardComputation(double[][][] W, double[] X) {
		
	}

	
	private void backwardComputation(double[][][] W, double[][] I, double[] X) {
		
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