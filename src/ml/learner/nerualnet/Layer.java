package ml.learner.nerualnet;

import ml.learner.neuralnet.initializers.WeightInitializer;
import ml.utils.Format;
import ml.utils.tracing.Trace;

public class Layer {
	private Net _net;
	private int _index;
	private int _inputs;
	private int _units;
	private WeightInitializer _weightInit;

	public Layer(int inputs) {
		this.init(inputs, inputs);
	}

	public Layer(int inputs, int units) {
		this.init(inputs, units);
	}

	private void init(int inputs, int units) {
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

	public void printWeightMatrix(double[][] W) {
		StringBuffer sb = new StringBuffer();

		sb.append("W[").append(this.index()).append("] = [\n");

		for (int j = 0; j < W.length; j++) {
			for (int i = 0; i < W[j].length; i++) {
				if (i > 0) {
					sb.append(" ");
				}
				sb.append(Format.sprintf("%1.4f", W[j][i]));
			}
			sb.append("\n");
		}
		sb.append("]\n");
		Trace.log(sb.toString());
	}

	public void initWeight(double[][][] W) {
		_weightInit.initializeWeights(this, W);
	}

}