package ml.learner.nerualnet;

public class Unit {
	public double[] weights;
	public double out;
	public double error;
	public double activation;
	
	private void computOutput(double[] input) {
		double out = 0;
		for (int i=0; i<input.length; i++) {
			out = out + input[i]*weights[i];
		}
	}
	
	public void computeActivation() {
		
	}
	
	public void computeError(double target) {
		
	}
	
	public void updateWeights() {
		
	}
	
}