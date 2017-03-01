package ml.utils;

import ml.utils.tracing.Trace;

public class Format {

	public static String matrix(Object v) {
		return matrix(v, 8);
	}


	public static String matrix(Object v, int decimal) {
		try {

			if (v instanceof double[]) {
				return format1d((double[]) v, decimal);
			}
			if (v instanceof double[][]) {
				return format2d((double[][]) v, decimal);
			}
			if (v instanceof double[][][]) {
				return format3d((double[][][]) v, decimal);
			}

			Trace.Error("Format.matrix(Object v): Unexpected data type ", v.toString());
		}
		catch (Exception ex) {
			Trace.Error("Format.matrix(Object v): Error formatting matrix ", v.toString(), "\nDetails:", ex.getMessage());
		}

		return "";
	}


	private static String format3d(double[][][] v, int decimal) {
		if (v == null) return "null";
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < v.length; i++) {
			if (i > 0) sb.append("\n");
			sb.append("[").append(i).append("]\n");
			sb.append(format2d((double[][]) v[i], decimal));
		}
		return sb.toString();
	}


	private static String format2d(double[][] v, int decimal) {
		if (v == null) return "null";
		StringBuffer sb = new StringBuffer();
		double m[][] = (double[][]) v;
		for (int i = 0; i < m.length; i++) {
			String row = format1d(m[i], 4);
			if (i > 0) sb.append("\n");
			sb.append(row);
		}
		return sb.toString();
	}


	private static String format1d(double[] v, int decimal) {
		if (v == null) return "null";
		StringBuffer sb = new StringBuffer();
		for (int j = 0; j < v.length; j++) {
			if (j > 0) sb.append(" ");
			String q = sprintf("%1." + decimal + "f", v[j]);
			sb.append(q);
		}
		return sb.toString();
	}


	public static String sprintf(String format, Object... values) {
		return String.format(format, values);
	}

}
