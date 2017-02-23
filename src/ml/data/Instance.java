package ml.data;

public class Instance {
	
	public double[] feature;
	public double[] target;
		
	public Instance() {
		
	}
	
	public Instance(int featureSize, int targetSize) {
		feature = new double[featureSize];
		target = new double[targetSize];
	}		
	
	
	private String formatVector(double[] v) {
		StringBuffer sb = new StringBuffer();
		
		sb.append("[");
		for (int i=0; i<v.length; i++) {
			if (i>0) sb.append(" ");
			sb.append(v[i]);
		}
		sb.append("]");
		
		return sb.toString();
	}
	
	
	public String toString() {	
		StringBuffer sb = new StringBuffer();
		
		sb.append("x = ").append(formatVector(feature));
		sb.append("\n");		
		sb.append("t = ").append(target);
		
		return sb.toString();
	}
}