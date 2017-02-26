package ml.learner.nerualnet.functions;

public class HyperbolicTangent implements Function {

	@Override
	public void compute(double[] x, double[] y) {
		for (int i = 0; i < x.length; i++) {
			y[i] = Math.tanh(x[i]);
		}
	}


	@Override
	public double compute(double x) {
		return Math.tanh(x);
	}


	@Override
	public void diff(double[] x, double[] y) {
		for (int i = 0; i < x.length; i++) {
			y[i] = Math.tanh(x[i]);
			y[i] = 1 - y[i] * y[i];
		}
	}


	@Override
	public double diff(double x) {
		double y = Math.tanh(x);
		return 1 - y * y;
	}


	@Override
	public void compute(Object... params) {
		// TODO Auto-generated method stub
		
	}


	@Override
	public void diff(Object... params) {
		// TODO Auto-generated method stub
		
	}

}