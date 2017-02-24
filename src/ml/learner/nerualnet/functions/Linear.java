package ml.learner.nerualnet.functions;

public class Linear implements Function {

	@Override
	public void compute(double[] x, double[] y) {
		for (int i = 0; i < x.length; i++) {
			y[i] =  Math.max(0, x[i]);
		}
	}

	@Override
	public double compute(double x) {
		return Math.max(0, x);
	}

	@Override
	public void computeDiff(double[] x, double[] y) {
		for (int i = 0; i < x.length; i++) {
			y[i] =  (x[i]<=0) ? 0 : 1;
		}
	}

	@Override
	public double computeDiff(double x) {
		return (x<=0) ? 0 : 1;
	}

}
