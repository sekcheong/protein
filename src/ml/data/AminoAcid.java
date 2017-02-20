package ml.data;

import java.util.HashMap;

public class AminoAcid {

	private static int total = 0;

	public int id = 0;

	// primary structure category value
	private int _first;

	// secondary structure category value
	private int _second;

	// primary structure labels, the 21st label is a made up
	// one and is used for padding purpose
	private static String[] _primaryLabels = { "A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F",
			"P", "S", "T", "W", "Y", "V", "?" };

	// secondary structure labels
	private static String[] _secondaryLabels = { "_", "h", "e" };

	// label to category map for primary structure
	private static HashMap<String, Integer> _primaryLabelToCat = new HashMap<String, Integer>();

	// label to category map for secondary structure
	private static HashMap<String, Integer> _secondaryLabelToCat = new HashMap<String, Integer>();

	// one hot encoding for the primary structure
	public static int[][] _primaryOneHot = new int[21][21];

	// one hot encoding for the secondary structure
	public static int[][] _secondaryOneHot = new int[3][3];

	public static AminoAcid padding = new AminoAcid();

	static {

		for (int i = 0; i < _primaryLabels.length; i++) {
			_primaryLabelToCat.put(_primaryLabels[i], i);
			_primaryOneHot[i][i] = 1;
		}

		for (int i = 0; i < _secondaryLabels.length; i++) {
			_secondaryLabelToCat.put(_secondaryLabels[i], i);
			_secondaryOneHot[i][i] = 1;
		}
	}

	public AminoAcid() {
		// use the 21st value to indicate unknown acid
		this._first = _primaryLabels.length - 1;
		this._second = 0;
		this.id = total;
		total++;
	}

	public AminoAcid(String code) {
		code = code.trim();
		String[] pieces = code.split("\\s+");
		this._first = _primaryLabelToCat.get(pieces[0].toUpperCase());
		this._second = _secondaryLabelToCat.get(pieces[1].toLowerCase());
		this.id = total;
		total++;
	}

	public int primary() {
		return _first;
	}

	public int secondary() {
		return _second;
	}

	public int[] primaryOneHot(int primaryCat) {
		return _primaryOneHot[primaryCat];
	}

	public int[] secondaryOneHot(int primaryCat) {
		return _secondaryOneHot[primaryCat];
	}

	public static String primaryLabel(int cat) {
		return _primaryLabels[cat];
	}

	public static String secondaryLabel(int cat) {
		return _secondaryLabels[cat];
	}

	public String toString() {
		return id + ": " + _primaryLabels[this._first] + " " + _secondaryLabels[this._second];
	}

}