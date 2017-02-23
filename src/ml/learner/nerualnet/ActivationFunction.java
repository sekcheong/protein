package ml.learner.nerualnet;

public interface ActivationFunction {
	public void layer(Layer layer);	
	public double[] compute(double[] potential);
	public double compute(double potential);
}