package ml.learner.nerualnet.functions;

public interface Function {	
	public void compute(double[] x, double[] y);
	public double compute(double x);
	public void computeDiff(double[] x, double[] y);
	public double computeDiff(double x);
}