package ml.learner.neuralnet.functions;

public class Sigmoid implements Function {

	@Override
	public void eval(double[] x, double[] y) {
		for (int i = 0; i < x.length; i++) {
			y[i] = 1 / (1 + Math.exp(-x[i]));
		}
	}


	@Override
	public double eval(double x) {
		return 1 / (1 + Math.exp(-x));
	}


	@Override
	public void diff(double[] x, double[] y) {
		for (int i = 0; i < x.length; i++) {
			y[i] = 1 / (1 + Math.exp(-x[i]));
			y[i] = y[i] * (1 - y[i]);
		}
	}


	@Override
	public double diff(double x) {
		double y = 1 / (1 + Math.exp(-x));
		return y * (1 - y);
	}


	@Override
	public void eval(Object... params) {
		// TODO Auto-generated method stub
		
	}


	@Override
	public void diff(Object... params) {
		// TODO Auto-generated method stub
		
	}

}
