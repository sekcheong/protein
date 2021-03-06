package ml.data;

import java.util.ArrayList;
import java.util.List;

import ml.utils.tracing.Trace;


public class DataSet {

	private List<List<Amino>> _dataSet;
	private List<List<Amino>> _proteins;
	private List<Amino> _paddings;
	private List<Instance> _instances = null;

	private int _windowSize;
	private int _size = 0;


	public DataSet(List<List<Amino>> proteins, int windowSize) throws Exception {

		if ((windowSize % 2) == 0) throw new Exception("DataSet(): windowSize must be a odd number.");

		_windowSize = windowSize;
		_size = 0;

		// creates the padding vector
		int paddingCount = _windowSize / 2;
		_paddings = new ArrayList<Amino>();
		for (int i = 0; i < paddingCount; i++) {
			_paddings.add(Amino.padding);
		}

		_proteins = proteins;
		for (List<Amino> p : proteins) {
			_size += p.size();
		}
	}


	public int size() {
		return _size;
	}


	private Instance createInstance(Amino[] window) {
		Instance inst = new Instance();
		int cols = Amino.primaryLabelCount();
		double[] features = new double[window.length * cols];
		double[] target = new double[Amino.secondaryLabelCount()];
		Amino center = window[window.length / 2];

		for (int i = 0; i < window.length; i++) {
			int[] v = window[i].primaryOneHot();
			for (int j = 0; j < cols; j++) {
				features[i * cols + j] = v[j];
			}
		}

		for (int i = 0; i < target.length; i++) {
			target[i] = center.secondaryOneHot()[i];
		}

		inst.features = features;
		inst.target = target;

		// logInstance(inst);
		return inst;
	}


	private List<Instance> createInstances() {
		Amino[] window = new Amino[_windowSize];
		List<Instance> insts = new ArrayList<Instance>();

		for (List<Amino> p : _proteins) {
			List<Amino> m = new ArrayList<Amino>();
			int size = p.size();
			m.addAll(_paddings);
			m.addAll(p);
			m.addAll(_paddings);
			for (int i = 0; i < size; i++) {
				for (int j = 0; j < _windowSize; j++) {
					window[j] = m.get(i + j);
				}
				Instance inst = createInstance(window);
				insts.add(inst);
			}
			// break;
		}
		// Trace.log("insts:", insts.size());
		return insts;
	}


	private void logWindow(Amino[] window) {
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < window.length; i++) {
			sb.append(window[i].toString()).append(", ");
		}
		Trace.log(sb);
	}


	private int findValue(double[] vector, double value, int offset, int length) {
		for (int i = 0; i < length; i++) {
			if (vector[offset + i] == value) return i;
		}
		return -1;
	}


	private void logInstance(Instance inst) {
		StringBuffer sb = new StringBuffer();
		String primary = null;
		int cols = Amino.primaryLabelCount();

		sb.append("[");

		for (int i = 0; i < _windowSize; i++) {
			int cat = findValue(inst.features, 1, i * cols, cols);
			if (cat > -1) {
				primary = Amino.primaryLabel(cat);
				sb.append(primary);
			}
			else {
				Trace.log("Error");
			}
		}

		int cat = findValue(inst.target, 1, 0, inst.target.length);
		String secondary = Amino.secondaryLabel(cat);

		sb.append("] = ").append(secondary);

		Trace.log(sb.toString());
	}


	public boolean verifyInstances() {
		List<Amino> src = new ArrayList<Amino>();

		for (List<Amino> a : _proteins) {
			src.addAll(a);
		}

		int cols = Amino.primaryLabelCount();
		int first = 0;
		int second = 0;
		for (Instance inst : this.instances()) {
			first = findValue(inst.features, 1, (_windowSize / 2) * cols, cols);
			second = findValue(inst.target, 1, 0, inst.target.length);
			// Trace.log(AminoAcid.primaryLabel(first) + " " +
			// AminoAcid.secondaryLabel(second));
			Amino acid = src.remove(0);
			if (acid.primary() != first || acid.secondary() != second) {
				Trace.TraceError("corrupted");
				return false;
			}
		}
		return true;
	}


	public Instance[] instances() {
		Instance[] ret;
		if (_instances == null) {
			List<Instance> insts = createInstances();
			_instances = insts;
		}
		ret = _instances.toArray(new Instance[_instances.size()]);
		return ret;
	}


	// put the 5th, 10th, 15th, …, 125th in your TUNE SET (counting from 1)
	// put the 6th, 11th, 16th, …, 126th in your TEST SET (counting from 1)
	// put the rest in your TRAIN set

	public DataSet[] Split() {
		List<List<Amino>> tune = new ArrayList<List<Amino>>();
		List<List<Amino>> test = new ArrayList<List<Amino>>();
		List<List<Amino>> train = new ArrayList<List<Amino>>();

		DataSet[] ret = new DataSet[3];

		int i = 1;
		for (List<Amino> p : _proteins) {
			if ((i % 5) == 0) {
				tune.add(p);
			}
			else if ((i % 5) == 1 && (i > 5)) {
				test.add(p);
			}
			else {
				train.add(p);
			}
			i++;
		}

		try {
			ret[0] = new DataSet(train, this._windowSize);
			ret[1] = new DataSet(tune, this._windowSize);
			ret[2] = new DataSet(test, this._windowSize);
		}
		catch (Exception ex) {
			ex.printStackTrace();
		}
		return ret;
	}
}