package ml.utils;

public class Format {
	
	public static String vector(double[] v) {
		StringBuffer sb = new StringBuffer();
		sb.append("[");
		for (int i = 0; i < v.length; i++) {
			if (i > 0) sb.append(" ");
			sb.append(v[i]);
		}
		sb.append("]");
		return sb.toString();
	}
	
	public static String sprintf(String format, Object... values) {
		return String.format(format, values);
	}
	
}
