package ml.learner.neuralnet;

import ml.learner.neuralnet.functions.*;
import ml.learner.neuralnet.initializers.DefaultWeightInitializer;
import ml.learner.neuralnet.initializers.WeightInitializer;

public class Layer {

	private NeuralNet _net;
	private WeightInitializer _weightInit = null;
	private Function _activationFunc = null;

	private int _index;
	private int _inputs;
	private int _units;

	private static Function _defaultActivationFunc = new Sigmoid();

	private static WeightInitializer _defaultWeightInitializer = new DefaultWeightInitializer();


	public Layer(int units, int inputs) {
		_units = units;
		_inputs = inputs;
	}


	public NeuralNet net() {
		return _net;
	}


	public void net(NeuralNet net) {
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


	public Layer weightInitializer(WeightInitializer init) {
		_weightInit = init;
		return this;
	}


	public WeightInitializer weightInitializer() {
		if (_weightInit == null) return _defaultWeightInitializer;
		return _weightInit;
	}


	public Layer activationFunction(Function func) {
		_activationFunc = func;
		return this;
	}


	public Function activationFunction() {
		if (_activationFunc == null) return _defaultActivationFunc;
		return _activationFunc;
	}


	@Override
	public String toString() {
		return "L[" + this.index() + "]: " + "inputs:" + this.inputs() + ", units:" + this.units();
	}

}