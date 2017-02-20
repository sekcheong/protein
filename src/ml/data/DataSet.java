package ml.data;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import ml.utils.tracing.Trace;

public class DataSet {

	private List<List<AminoAcid>> _dataSet;
	private List<List<AminoAcid>> _proteins;
	private List<AminoAcid> _paddings;
	private List<Instance> _instances = null;

	private int _windowSize;
	private int _size = 0;

	
	public DataSet(List<List<AminoAcid>> proteins, int windowSize) throws Exception {

		if ((windowSize % 2) == 0) throw new Exception("DataSet(): windowSize must be a odd number.");

		_windowSize = windowSize;
		_size = 0;

		// creates the padding vector
		int paddingCount = _windowSize / 2;
		_paddings = new ArrayList<AminoAcid>();
		for (int i = 0; i < paddingCount; i++) {
			_paddings.add(AminoAcid.padding);
		}

		_proteins = proteins;
		for (List<AminoAcid> p : proteins) {
			_size += p.size();
		}
	}

	
	public int size() {
		return _size;
	}

	
	private Instance createInstance(AminoAcid[] window) {
		Instance inst = new Instance();
		int cols = AminoAcid.primaryLabelCount();
		double[] features = new double[window.length * cols];
		double[] target = new double[AminoAcid.secondaryLabelCount()];
		AminoAcid center = window[window.length / 2];

		for (int i = 0; i < window.length; i++) {
			int[] v = window[i].primaryOneHot();
			for (int j = 0; j < cols; j++) {
				features[i * cols + j] = v[j];
			}
		}

		for (int i = 0; i < target.length; i++) {
			target[i] = center.secondaryOneHot()[i];
		}

		inst.feature = features;
		inst.target = target;

		//logInstance(inst);
		return inst;
	}
	

	private List<Instance> createInstances() {
		AminoAcid[] window = new AminoAcid[_windowSize];
		List<Instance> insts = new ArrayList<Instance>();

		for (List<AminoAcid> p : _proteins) {
			List<AminoAcid> m = new ArrayList<AminoAcid>();
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
			//break;
		}
		//Trace.log("insts:", insts.size());
		return insts;
	}

	
	private void logWindow(AminoAcid[] window) {
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
		int cols = AminoAcid.primaryLabelCount();

		sb.append("[");

		for (int i = 0; i < _windowSize; i++) {
			int cat = findValue(inst.feature, 1, i * cols, cols);
			if (cat > -1) {
				primary = AminoAcid.primaryLabel(cat);
				sb.append(primary);
			}
			else {
				Trace.log("Error");
			}
		}

		int cat = findValue(inst.target, 1, 0, inst.target.length);
		String secondary = AminoAcid.secondaryLabel(cat);

		sb.append("] = ").append(secondary);

		Trace.log(sb.toString());
	}

	
	public boolean verifyInstances() {
		List<AminoAcid> src = new ArrayList<AminoAcid>();

		for (List<AminoAcid> a : _proteins) {
			src.addAll(a);
		}

		int cols = AminoAcid.primaryLabelCount();
		int first = 0;
		int second = 0;
		for (Instance inst : this.getInstances()) {
			first = findValue(inst.feature, 1, (_windowSize / 2) * cols, cols);
			second = findValue(inst.target, 1, 0, inst.target.length);
			//Trace.log(AminoAcid.primaryLabel(first) + " " + AminoAcid.secondaryLabel(second));
			AminoAcid acid = src.remove(0);
			if (acid.primary()!=first || acid.secondary()!=second) {
				Trace.Error("Corrupted");
				return false;
			}
		}
		return true;
	}

	
	public Instance[] getInstances() {
		Instance[] ret;
		if (_instances == null) {
			List<Instance> insts = createInstances();
			_instances = insts;
		}
		ret = _instances.toArray( new Instance[_instances.size()]);
		return ret;
	}
	

	public DataSet[] Split(double ratio) {
		List<List<AminoAcid>> src = new ArrayList<List<AminoAcid>>(_proteins);
		List<List<AminoAcid>> dest = new ArrayList<List<AminoAcid>>();
		DataSet[] ret = new DataSet[2];
		
		int target = (int) (_size * ratio);
		int size = 0;
		//Trace.log("target:", target);

		while (true) {
			List<AminoAcid> p = src.remove(0);
			dest.add(p);
			size = size + p.size();
			if (size >= target) break;
		}
		
		try {
			DataSet s1 = new DataSet(dest, this._windowSize);
			DataSet s2 = new DataSet(src, this._windowSize);
			ret[0] = s1;
			ret[1] = s2;
			//Trace.log("s1:", s1.size());
			//Trace.log("s2:", s2.size());
		}
		catch (Exception ex) {
			ex.printStackTrace();
		}
		return ret;
	}
}