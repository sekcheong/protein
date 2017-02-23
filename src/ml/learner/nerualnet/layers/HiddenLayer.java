package ml.learner.nerualnet.layers;

public class HiddenLayer extends AbstractLayer {
	private AbstractLayer _lastLayer;
	private int _units;
	
	public HiddenLayer(int inputs, int units) {
		super(inputs);
		_units = units;
	}
	
	public int units() {
		return _units;
	}
}