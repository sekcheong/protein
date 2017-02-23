package ml.learner.nerualnet;

import ml.learner.nerualnet.layers.AbstractLayer;

public class Net {

	public double[][][] weights = null;
	private int _currentLayer = 0;

	public Net(int layers) {
		this.weights = new double[layers][][];
	}

	public void AddLayer(AbstractLayer layer) {
		//weights[_currentLayer] = new double[layer.inputs()][layer.units()];
		_currentLayer++;
	}
}
