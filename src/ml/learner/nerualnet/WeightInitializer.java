package ml.learner.nerualnet;

public interface WeightInitializer {	
	public void layer(Layer layer);
	public void initialize();
}