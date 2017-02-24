package ml.learner.nerualnet.functions;

public interface Function {	
	public void compute(double[] x, double[] y);
	public double compute(double x);
	public void diff(double[] x, double[] y);
	public double diff(double x);
}