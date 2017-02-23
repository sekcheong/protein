package ml.learner.nerualnet;

import ml.learner.nerualnet.layers.*;

public class Net {

	// W[L,j,i] are weight matrices where L is the layer number, j is the
	// j_th neuron of L, and i is the i_th neuron of layer L-1
	private double[][][] W = null;

	// I[L,j] are vectors whose element denote weighted input of j_th
	// neuron of layer L
	private double[][] I = null;

	// Y[L,j] are vectors whose elements denote the output of the j-th
	// neuron of layer the
	private double[][] Y = null;

	// the previous error
	private double oldE;

	// the new error
	private double E;

	// private List<AbstractLayer> _layers = new ArrayList<AbstractLayer>();

	private AbstractLayer[] _layers = new AbstractLayer[5];
	private int _layerCount = 0;

	private int _epoch;

	// the target error rate
	private double _epsilon;

	// the learning rate
	private double _eta;

	public Net(int layers) {
		W = new double[layers][][];
	}
	

	private void addLayer(AbstractLayer layer) {

		if (_layerCount == _layers.length) {
			this.expandLayers();
		}
		
		_layers[_layerCount] = layer;
		_layerCount++;
		layer.net(this);
	}

	
	private void expandLayers() {
		AbstractLayer[] newLayers = new AbstractLayer[_layerCount * 2];
		for (int i = 0; i < _layers.length; i++) {
			newLayers[i] = _layers[i];
		}
		_layers = newLayers;
	}
	
	
	private void trimLayers() {
		if (_layerCount==_layers.length) return;
		AbstractLayer[] newLayers = new AbstractLayer[_layerCount];
		for (int i = 0; i < _layerCount; i++) {
			newLayers[i] = _layers[i];
		}
		_layers = newLayers;
	}
	
	
	public AbstractLayer[] layers() {
		trimLayers();
		return _layers;
	}
	

	public void inputLayer(InputLayer layer) {
		addLayer(layer);
	}

	
	public void addHiddenLayer(HiddenLayer layer) {
		addLayer(layer);
	}

	
	public void outputLayer(OutputLayer layer) {
		addLayer(layer);
	}

	
	public void initialize() {

	}

}