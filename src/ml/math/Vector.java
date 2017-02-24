package ml.math;

import ml.utils.tracing.Trace;

public class Vector {

	public static void wtx(double[][] w, double[] x, double[] y) {
		Trace.log("(" + w.length + "," + w[0].length + ")");
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

	public static void sub(double[] u, double[] v, double[] y) {
		for (int i = 0; i < u.length; i++) {
			y[i] = u[i] - v[i];
		}
	}

	public static void add(double[] u, double[] v, double[] y) {
		for (int i = 0; i < u.length; i++) {
			y[i] = u[i] + v[i];
		}
	}

	public static double sum(double[] u) {
		double y = 0;
		for (int i = 0; i < u.length; i++) {
			y = y + u[i];
		}
		return y;
	}

	public static void square(double[] u, double[] y) {
		for (int i = 0; i < u.length; i++) {
			y[i] = u[i] * u[i];
		}
	}

	public static void scale(double[] u, double c) {
		for (int i = 0; i < u.length; i++) {
			u[i] = u[i] * c;
		}
	}

}