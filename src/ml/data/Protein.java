package ml.data;

import java.util.ArrayList;
import java.util.List;

public class Protein {
	
	private List<AminoAcid> _aminos = new ArrayList<AminoAcid>();
	
	public int length() {		
		return _aminos.size();		
	}	
	
}