package ml.learner.nerualnet;

import java.text.DecimalFormat;
import java.text.NumberFormat;

import ml.learner.nerualnet.config.WeightInitializer;
import ml.utils.tracing.Trace;

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

	public void printWeightMatrix() {
		NumberFormat formatter = new DecimalFormat("#0.0000");
		StringBuffer sb = new StringBuffer();
		double[][][] W = _net.W();
		int l = this.index();
		
		sb.append("W[").append(this.index()).append("] = [\n");		
		
		for (int j=0; j<W[l].length; j++) {
			for (int i=0; i<W[l][j].length; i++) {
				if (i==0) {
					sb.append(formatter.format(W[l][j][i]));
				}
				else {
					sb.append(" ").append(formatter.format(W[l][j][i]));
				}
			}
			sb.append("\n");
		}
		sb.append("]\n");
		Trace.log(sb.toString());
	}
	
	
}