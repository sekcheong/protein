package ml.data;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import ml.utils.tracing.Trace;

public class DataSet implements Iterable<Instance> {

	private class InstanceIterator implements Iterator<Instance> {

		// the window position of the amino acids in the current protein
		private int _currWindow = 0;
		// the window size
		private int _windowSize = 0;

		private int _count = 0;
		private int _total = 0;

		private List<AminoAcid> _currProtein;
		private int _currProteinPos = 0;
		private int _currAminoPos = 0;
		private Instance _instance;

		private List<List<AminoAcid>> _proteins;
		private List<AminoAcid> _paddings;

		public InstanceIterator(List<List<AminoAcid>> proteins, List<AminoAcid> paddings, Instance inst, int totalInstances) {
			_proteins = proteins;
			_windowSize = paddings.size() * 2 + 1;
			_paddings = paddings;
			_total = totalInstances;
			_instance = inst;
		}

		@Override
		public boolean hasNext() {
			return (_count < _total);
		}

		@Override
		public Instance next() {

			if (_currProtein == null) {
				_currProtein = new ArrayList<AminoAcid>();
				_currProtein.addAll(_paddings);
				_currProtein.addAll(_proteins.get(_currProteinPos));
				_currProtein.addAll(_paddings);
				_currProteinPos++;
				_currAminoPos = 0;
			}

			Instance inst = getOneInstance(_currAminoPos, _currProtein, _windowSize);
			_currAminoPos++;

			if (_currAminoPos == _proteins.get(_currProteinPos).size()) {
				_currProtein = null;
			}
			_count++;
			return inst;
		}

		private Instance getOneInstance(int index, List<AminoAcid> protein, int windowSize) {
			int cols = AminoAcid.primaryLabelCount();
			Instance inst = new Instance(windowSize * cols);
			for (int i = 0; i < windowSize; i++) {
				AminoAcid a = protein.get(index+i);
				int[] oneHot = a.primaryOneHot();
				for (int j=0; j<cols; j++) {
					inst.features[i*cols+j] = oneHot[j];
				}
			}
			inst.target = protein.get(index + windowSize / 2).secondary();
			return inst;
		}

	}

	private int _windowSize;
	private int _instanceCount = 0;

	private List<List<AminoAcid>> _dataSet;
	private List<List<AminoAcid>> _proteins;
	private List<AminoAcid> _paddings;
	private Instance _instance;

	public DataSet(List<List<AminoAcid>> proteins, int windowSize) throws Exception {

		if ((windowSize % 2) == 0) throw new Exception("DataSet(): windowSize must be a odd number.");

		_windowSize = windowSize;
		_instanceCount = 0;

		// creates the padding vector
		int paddingCount = _windowSize / 2;
		_paddings = new ArrayList<AminoAcid>();
		for (int i = 0; i < paddingCount; i++) {
			_paddings.add(AminoAcid.padding);
		}

		_proteins = proteins;
		for (List<AminoAcid> p : proteins) {
			List<AminoAcid> m = new ArrayList<AminoAcid>();
			_instanceCount += p.size();
		}

		_instance = new Instance(_windowSize * AminoAcid.primaryLabelCount());

	}

	public List<DataSet> Split(double ratio) {

		List<List<AminoAcid>> src = new ArrayList<List<AminoAcid>>(_proteins);
		List<List<AminoAcid>> dest = new ArrayList<List<AminoAcid>>();

		// Collections.shuffle(src);

		int target = (int) (_instanceCount * ratio);
		int size = 0;
		Trace.log("target:", target);

		while (true) {
			List<AminoAcid> p = src.remove(0);
			dest.add(p);
			size = size + p.size();
			if (size >= target) break;
		}

		List<DataSet> s = new ArrayList<DataSet>();
		try {
			DataSet s1 = new DataSet(dest, this._windowSize);
			DataSet s2 = new DataSet(src, this._windowSize);
			Trace.log("s1:", s1.instanceCount());
			Trace.log("s2:", s2.instanceCount());
			s.add(s1);
			s.add(s2);
		}
		catch (Exception ex) {
			ex.printStackTrace();
		}
		return s;
	}

	public int size() {
		return _proteins.size();
	}

	public int instanceCount() {
		return _instanceCount;
	}

	@Override
	public Iterator<Instance> iterator() {
		return new InstanceIterator(_proteins, _paddings, _instance, _instanceCount);
	}

}