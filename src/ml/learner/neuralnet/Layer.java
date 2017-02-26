package ml.learner.neuralnet;

import ml.learner.neuralnet.functions.Function;
import ml.learner.neuralnet.initializers.WeightInitializer;

public class Layer {

	private NeuralNet _net;
	private WeightInitializer _weightInit;
	private Function _activationFunc;

	private int _index;
	private int _inputs;
	private int _units;


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

		return _weightInit;
	}


	public Layer activationFunction(Function func) {
		_activationFunc = func;
		return this;
	}


	public Function activationFunction() {
		return _activationFunc;
	}


	@Override
	public String toString() {
		return "L[" + this.index() + "]: " + "inputs:" + this.inputs() + ", units:" + this.units();
	}

}