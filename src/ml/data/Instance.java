package ml.data;

public class Instance {
	
	//feature vector
	public double[] features;
	
	public Instance(int size) {
		features = new double[size];
	}
	
	public double target;
}