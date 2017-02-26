package ml.learner.nerualnet.functions;

import java.util.Random;

public class NosyLinear implements Function {
	private Random _r = new Random();
	private double _mean;
	private double _variance;


	public NosyLinear() {
		_mean = 0;
		_variance = 0.3;
	}


	public NosyLinear(double mean, double variance) {
		_mean = mean;
		_variance = variance;
	}


	@Override
	public void compute(double[] x, double[] y) {
		double noise = _r.nextGaussian() * Math.sqrt(_variance) + _mean;
		for (int i = 0; i < x.length; i++) {
			y[i] = Math.max(0, x[i] + noise);
		}
	}


	@Override
	public double compute(double x) {
		double noise = _r.nextGaussian() * Math.sqrt(_variance) + _mean;
		return Math.max(0, x + noise);
	}


	@Override
	public void diff(double[] x, double[] y) {
		for (int i = 0; i < x.length; i++) {
			y[i] = (x[i] <= 0) ? 0 : 1;
		}
	}


	@Override
	public double diff(double x) {
		return (x <= 0) ? 0 : 1;
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
