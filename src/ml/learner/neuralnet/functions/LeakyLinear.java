package ml.learner.neuralnet.functions;

public class LeakyLinear implements Function {

	private double _slope;


	public LeakyLinear() {
		_slope = 0.01;
	}


	public LeakyLinear(double slope) {
		_slope = slope;
	}


	@Override
	public void eval(double[] x, double[] y) {
		for (int i = 0; i < x.length; i++) {
			y[i] = (x[i] < 0) ? x[i] * _slope : x[i];
		}
	}


	@Override
	public double eval(double x) {
		return (x < 0) ? x * _slope : x;
	}


	@Override
	public void diff(double[] x, double[] y) {
		for (int i = 0; i < x.length; i++) {
			y[i] = (x[i] < 0) ? -_slope : 1;
		}
	}


	@Override
	public double diff(double x) {
		return (x < 0) ? -_slope : 1;
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
