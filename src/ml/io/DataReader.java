package ml.io;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import ml.data.AminoAcid;
import ml.utils.console.Console;
import ml.utils.tracing.Trace;

public class DataReader {

	private BufferedReader _buffReader = null;

	public DataReader(String fileName) throws Exception {
		FileReader freader = new FileReader(fileName);
		_buffReader = new BufferedReader(freader);
	}

	public List<List<AminoAcid>> Read() throws Exception {

		List<List<AminoAcid>> proteins = new ArrayList<List<AminoAcid>>();
		List<AminoAcid> protein = null;
		
		int lineNo = 0;
		
		while (true) {
						
			String line = _buffReader.readLine();
			lineNo++;
			
			if (line == null) {
				if (protein != null) {
					proteins.add(protein);
				}
				break;
			}
			
			line = line.trim();
			
			//skip comment line
			if (line.startsWith("#") || line.startsWith("//")) {
				continue;
			}

			//starts of a new protein sequence
			if (line.startsWith("<>")) {				
				if (protein != null) {
					proteins.add(protein);
				}
				protein = new ArrayList<AminoAcid>();
				continue;
			}
			
			//skip the end of protein mark
			if (line.toUpperCase().startsWith("END")|| line.toUpperCase().startsWith("<END") || line.toUpperCase().startsWith("<END>")) {
				continue;
			}

			// parse the primary and secondary structure line
			if (line.length() > 0) {
				try {
					AminoAcid amino = new AminoAcid(line);
					protein.add(amino);
				}
				catch (Exception ex) {
					Console.Error("Unable to process sequence at line ", lineNo, ":", line);
				}
			}
		}		
		
		int count = 0;
		for (List<AminoAcid> p : proteins) {
			count += p.size();
		}
		
		Trace.log("Total proteins:", proteins.size());
		Trace.log("Total amino   :", count);
		
		return proteins;
	}
}
