package ml.learner.neuralnet;

public class Neuron {
	private Layer _layer;
	private int _index;
	private int _layerIndex;
	
	
	public Neuron(Layer l) {
		_layer = l;
		_layerIndex = l.index();
	}
	
	
	public int index() {
		return _index;
	}
	
	
	public int layerIndex() {
		return _layerIndex;
	}
	
}