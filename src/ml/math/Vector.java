package ml.math;

public class Vector {

	public static void wtx(double[] y, double[][] w, double[] x) {
		for (int i = 0; i < w.length; i++) {
			y[i] = dot(w[i], x);
		}
	}
	
	
	public static double dot(double[] u, double[] v) {
		double y = 0;
		for (int i = 0; i < u.length; i++) {
			y = y + u[i] * v[i];
		}
		return y;
	}

	
	public static void sub(double[] y, double[] u, double[] v) {
		for (int i = 0; i < u.length; i++) {
			y[i] = u[i] - v[i];
		}
	}
	
	
	public static void sum(double[] y, double[] u, double[] v) {
		for (int i = 0; i < u.length; i++) {
			y[i] = u[i] + v[i];
		}
	}
	
	
	public static void scale(double[] u, double c) {
		for (int i=0; i<u.length; i++) {
			u[i] = u[i]*c;
		}
	}

}