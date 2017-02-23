package ml.learner.nerualnet;

public class Net {

	// W[L,j,i] are weight matrices where L is the layer number, j is the
	// j_th neuron of L, and i is the i_th neuron of layer L-1
	private double[][][] W = null;

	// I[L,j] are vectors whose element denote weighted input of j_th
	// neuron of layer L
	private double[][] I = null;

	// Y[L,j] are vectors whose elements denote the output of the j_th
	// neuron of layer the
	private double[][] Y = null;

	// the previous error
	private double oldE;

	// the new error
	private double E;

	private int _epoch;

	// the target error rate
	private double _epsilon;

	// the learning rate
	private double _eta;

	private Layer[] _layers;
	private int _layerCount;
	
	
	public Net(int layers) {
		_layers = new Layer[layers];
		_layerCount=0;
	}
	

	public void addLayer(Layer layer) {
		_layers[_layerCount] = layer;
		layer.index(_layerCount);
		layer.net(this);
		_layerCount++;
	}

	
	public Layer[] layers() {
		return _layers;
	}

	
	public double[][][] W() {
		return this.W;
	}
	
	
	public void initialize() {
		W = new double[_layers.length][][];
		W[0] = new double[_layers[0].inputs()][];		
		for (int i=1; i<_layers.length; i++) {
			W[i] = new double[_layers[i].units()][_layers[i-1].units()];
			_layers[i-1].initWeight();
		}		
	}

}