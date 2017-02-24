package ml.learner.nerualnet;

import java.util.ArrayList;
import java.util.List;

public class Net {

	private static double BIAS = -1.0;
	// W[L,j,i] are weight matrices where L is the layer number, j is the
	// j_th neuron of L, and i is the i_th neuron of layer L-1
	public double[][][] W = null;

	// I[L,j] are vectors whose element denote weighted input of j_th
	// neuron of layer L
	private double[][] I = null;

	// Y[L,j] are vectors whose elements denote the output of the j_th
	// neuron of layer L
	private double[][] Y = null;

	// the error for previous example
	private double oldE;

	// the error for the current example
	private double E;

	// the cumulative error rate after k_th example
	private double Em;

	private int _epoch;

	// the target error rate
	private double _epsilon;

	// the learning rate
	private double _eta;

	private Layer[] _layers;

	private List<Layer> _tempLayers = new ArrayList<Layer>();
	
	
	public Net() { }

	
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
	

	public double[][][] W() {
		return this.W;
	}

	
	private void setBias(double[][][] w) {
		for (int l=1; l<w.length; l++) {
			for (int j=0; j<w[l].length; j++) {
				w[l][j][0] = BIAS;
			}
		}
	}
	
	
	public void initialize() {
		
		Layer layres[] = this.layers();
		
		// creates the weight matrices and set the initial weights
		W = new double[layres.length][][];
		I = new double[layres.length][];
		Y = new double[layres.length][];
		
		// create and initialize hidden layer weight matrices
		for (int l = 1; l < layres.length; l++) {
			
			W[l] = new double[layres[l].units()][];
			I[l] = new double[layres[l].units()];
			Y[l] = new double[layres[l].units()];
			
			for (int j = 0; j < layres[l].units(); j++) {
				W[l][j] = new double[layres[l - 1].units()];
			}
			
			layres[l].initWeight();
		}
		
		setBias(W);
		
		for (int l = 1; l < layres.length; l++) {
			layres[l].printWeightMatrix();
		}
	}

}