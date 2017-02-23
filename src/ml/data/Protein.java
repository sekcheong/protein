package ml.data;

import java.util.ArrayList;
import java.util.List;

public class Protein {
	
	private List<Amino> _aminos = new ArrayList<Amino>();
	
	public int length() {		
		return _aminos.size();		
	}
	
	public Amino[] aminoAcids() {
		return _aminos.toArray(new Amino[_aminos.size()]);
	}
	
}