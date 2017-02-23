package ml.learner.nerualnet;


public class Layer {
		
	public double output[] = null;
	private Net _net;
	private int _units;
	private int _inputs;
	
	private WeightInitializer _weightInitializer = null;
	
	
	public Layer(int units, int inputs) {		
		_units = units;
		_inputs = inputs;
	}
	
	
	public void weighttInitializer(WeightInitializer wieghtInit) {
		wieghtInit.layer(this);
	}
			
	
	public void activationFunction(ActivationFunction activateFunc) {
		
	}
	
	
	public Net net() {
		return _net;
	}
	
	
	public void net(Net net) {
		_net = net;
	}
	
	
	public int units() {
		return _units;
	}
	
	
	public int inputs() {
		return _inputs;
	}
}