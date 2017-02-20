
import java.util.List;

import ml.data.AminoAcid;
import ml.data.DataSet;
import ml.data.Instance;
import ml.io.DataReader;
import ml.learner.nerual.Neuron;
import ml.utils.misc.DataShuffler;
import ml.utils.tracing.Trace;

public class Program {

	public static void main(String[] args)  {				
		Trace.enabled = true;
		
		DataReader reader = null;
		try {
			reader = new DataReader(args[0]);
			List<List<AminoAcid>> val = reader.Read();
			DataSet dataSet = new DataSet(val, 17);			
			List<DataSet> subSets = dataSet.Split(0.2);
			
			for (Instance i:dataSet) {
				Trace.log(i);
			}
			
		}
		catch (Exception ex) {
			ex.printStackTrace();
		}

//		System.out.println("Hello!");
//		Neuron n = new Neuron();
//		n.ComputeError();
	}

}


//BufferedReader _buffReader = null;
//String _fileName = args[0];
//
//try {
//	FileReader freader = new FileReader(_fileName);
//	_buffReader = new BufferedReader(freader);
//}
//catch (Exception ex) {
//	System.out.println("Error:" + ex.getMessage());
//}
//
//List<List<String>> examples = new ArrayList<List<String>>();
//List<String> protein = null;
//
//int lineNo = 0;
//int count = 0;
//try {
//	while (true) {
//		String line;
//
//		line = _buffReader.readLine();
//		lineNo++;
//		if (line == null) {
//			if (protein != null) {
//				examples.add(protein);
//			}
//			break;
//		}
//		line = line.trim();
//		if (line.startsWith("#")) {
//			continue;
//		}
//		if (line.startsWith("<>")) {
//			count++;
//			if (protein != null) {
//				examples.add(protein);
//			}
//			protein = new ArrayList<String>();
//		}
//		else if (line.toUpperCase().startsWith("END") || line.toUpperCase().startsWith("<END>")) {
//			// examples.add(features);
//			// features = null;
//		}
//		else {
//			if (line.length() > 0) {
//				protein.add(line);
//			}
//			else {
//				System.out.println("Line: " + lineNo + " is blank.");
//			}
//		}
//	}
//
//	int total = 0;
//	for (int i = 0; i < examples.size(); i++) {
//		System.out.println(i + ": " + examples.get(i).size() + examples.get(i));
//		total = total + examples.size();
//	}
//
//	System.out.println("#" + examples.size() + ": " + count + " : " + total);
//}
//catch (Exception ex) {
//	ex.printStackTrace();
//}
