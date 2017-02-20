package ml.utils.misc;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.List;
import ml.data.AminoAcid;
import ml.io.DataReader;
import ml.learner.nerual.Neuron;

public class DataShuffler {

	public static void ShuffleProteinSequences(String srcFile, String destFile) {
		
		BufferedWriter writer = null;
		FileWriter file = null;		
		DataReader reader = null;
		
		try {
			
			reader = new DataReader(srcFile);
			List<List<AminoAcid>> val = reader.Read();
			file = new FileWriter(destFile);
			writer = new BufferedWriter(file);
			
			for (List<AminoAcid> p:val) {
				writer.write("<>");
				writer.newLine();
				for (AminoAcid a:p) {
					writer.write(a.toString());
					writer.newLine();
				}
				writer.write("<end>");
				writer.newLine();
			}
			writer.flush();			
		}
		catch (Exception ex) {			
			ex.printStackTrace();
		}
	}
}
