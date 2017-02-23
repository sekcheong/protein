package ml.learner.nerualnet.layers;

import ml.learner.nerualnet.Net;
import ml.learner.nerualnet.WeightInitializer;
import ml.learner.nerualnet.activation.ActivationFunction;

public abstract class AbstractLayer {	
	private Net _net;		
	private WeightInitializer _weightInitializer = null;
	public ActivationFunction g = null;
	
	private int _index;
	private int _inputs;
	
	
	public AbstractLayer(int inputs) {
		_inputs = inputs;
	}
	
	
	public void weightInitializer(WeightInitializer wieghtInit) {
		_weightInitializer = wieghtInit;
	}
	
	
	public WeightInitializer weightInitializer() {
		return _weightInitializer;		
	}
	
	
	public void activationFunction(ActivationFunction activateFunc) {
		this.g = activateFunc;
	}
	
	
	public Net net() {
		return _net;
	}
	
	
	public void net(Net net) {
		_net = net;
	}
	
	
	public int index() {
		return _index;
	}
	

	public void index(int index) {
		_index = index;
	}
	
}