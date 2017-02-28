package ml.learner.neuralnet.functions;

public interface Function {	
	public void eval(double[] x, double[] y);
	public void eval(Object...params);
	public double eval(double x);
	public void diff(double[] x, double[] y);
	public void diff(Object...params);
	public double diff(double x);
}