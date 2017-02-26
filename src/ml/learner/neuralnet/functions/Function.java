package ml.learner.neuralnet.functions;

public interface Function {	
	public void compute(double[] x, double[] y);
	public void compute(Object...params);
	public double compute(double x);
	public void diff(double[] x, double[] y);
	public void diff(Object...params);
	public double diff(double x);
}