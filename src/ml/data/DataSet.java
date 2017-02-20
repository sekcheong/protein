package ml.data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import ml.utils.tracing.Trace;

public class DataSet implements Iterable<Instance> {

	private class InstanceIterator implements Iterator<Instance> {

		private int _protenCount = 0;
		private int _aminoCount = 0;
		private int _windowSize =0;
		private List<List<AminoAcid>> _proteins;

		public InstanceIterator(List<List<AminoAcid>> proteins, int windowSize) {
			_windowSize = windowSize;
			_proteins = proteins;
		}

		@Override
		public boolean hasNext() {
			if (_proteins.size() < 1) return false;

			if (_protenCount == _proteins.size()) {
				if (_aminoCount == _proteins.get(_proteins.size() - 1).size()) {
					return false;
				}
			}

			return true;
		}

		@Override
		public Instance next() {
			// TODO Auto-generated method stub
			return null;
		}

	}
	
	
	private int _windowSize = 1;
	private List<List<AminoAcid>> _proteins;
	private int _instanceCount = 0;

	
	public DataSet(List<List<AminoAcid>> proteins, int windowSize) {
		_windowSize = windowSize;
		_proteins = proteins;
		_instanceCount = 0;
		
		for (List<AminoAcid> p : _proteins) {
			_instanceCount += p.size();
		}
	}
	

	public List<DataSet> Split(double ratio) {

		List<List<AminoAcid>> src = new ArrayList<List<AminoAcid>>(_proteins);
		List<List<AminoAcid>> dest = new ArrayList<List<AminoAcid>>();

		//Collections.shuffle(src);

		int target = (int) (_instanceCount * ratio);
		int size = 0;
		Trace.log("target:", target);

		while (true) {
			List<AminoAcid> p = src.remove(0);
			dest.add(p);			
			size = size + p.size();
			if (size >= target) break;
		}

		DataSet s1 = new DataSet(dest, this._windowSize);
		DataSet s2 = new DataSet(src, this._windowSize);

		Trace.log("s1:", s1.instanceCount());
		Trace.log("s2:", s2.instanceCount());

		List<DataSet> s = new ArrayList<DataSet>();
		s.add(s1);
		s.add(s2);

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
		return new InstanceIterator(_proteins, _windowSize);
	}

}