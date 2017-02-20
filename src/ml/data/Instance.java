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
	
	public String toString() {	
		StringBuffer sb = new StringBuffer();
		sb.append("[");
		for (int i=0; i<feature.length; i++) {
			sb.append(feature[i]).append(" ");
		}
		sb.append("]");
		sb.append("=").append(target);
		return sb.toString();
	}
}