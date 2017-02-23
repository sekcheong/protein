package ml.learner.nerualnet.layers;

public class HiddenLayer extends AbstractLayer {
	private AbstractLayer _lastLayer;
	
	public HiddenLayer(int units, int inputs) {
		super(units, inputs);		
	}
	
}
