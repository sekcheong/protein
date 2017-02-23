package ml.learner.nerualnet;

public class Neuron {
	
	private int _index;
	
	public Neuron(int index) {
		_index=index;
	}
	
	private double computOutput() {
		return 0;
	}

	public double computeActivation() {
		return 0;
	}

	public double computeError(double[] target) {
		return 0;
	}

	public void updateWeights() {
		
	}
	
	public int index() {
		return _index;
	}

}