package ml.learner.nerualnet;

import ml.learner.nerualnet.config.WeightInitializer;

public class Layer {
	private Net _net;
	private int _index;
	private int _inputs;
	private int _units;
	private WeightInitializer _weightInit;

	public Layer(int inputs, int units) {
		_inputs = inputs;
		_units = units;
	}

	public Net net() {
		return _net;
	}

	public void net(Net net) {
		_net = net;
	}

	public int index() {
		return _index;
	}

	public void index(int index) {
		_index = index;
	}

	public int inputs() {
		return _inputs;
	}

	public int units() {
		return _units;
	}

	public void weightInitializer(WeightInitializer init) {
		_weightInit = init;
	}

	public void initWeight() {		
		_weightInit.initializeWeights(this);
	}

}