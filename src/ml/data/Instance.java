package ml.data;

import ml.utils.Format;

public class Instance {
	
	public double[] features;
	public double[] target;
		
	public Instance() {
		
	}
	
	
	public Instance(int featureSize, int targetSize) {
		features = new double[featureSize];
		target = new double[targetSize];
	}		
	
	public String toString() {	
		StringBuffer sb = new StringBuffer();
		
		sb.append("x = ").append(Format.vector(features));
		sb.append("\n");		
		sb.append("t = ").append(target);
		
		return sb.toString();
	}
}