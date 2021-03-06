package ml.learner.neuralnet.functions;

public class Linear implements Function {

	@Override
	public void eval(double[] x, double[] y) {
		for (int i = 0; i < x.length; i++) {
			y[i] = Math.max(0, x[i]);
		}
	}


	@Override
	public double eval(double x) {
		return Math.max(0, x);
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
	public void eval(Object... params) {
		// TODO Auto-generated method stub
		
	}


	@Override
	public void diff(Object... params) {
		// TODO Auto-generated method stub
		
	}

}
