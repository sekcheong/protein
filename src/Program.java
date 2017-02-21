
import java.util.List;

import ml.data.AminoAcid;
import ml.data.DataSet;
import ml.data.Instance;
import ml.io.DataReader;
import ml.learner.nerualnet.Neuron;
import ml.utils.misc.DataShuffler;
import ml.utils.tracing.Trace;

public class Program {

	public static void main(String[] args) {
		Trace.enabled = true;

		DataReader reader = null;
		Instance[] train = null;
		Instance[] tune = null;		

		try {
			
			long startTime = System.currentTimeMillis();
						
			reader = new DataReader(args[0]);			
			List<List<AminoAcid>> val = reader.Read();
			
			DataSet dataSet = new DataSet(val, 17);			
			DataSet[] subSets = dataSet.Split(0.2);
			train = subSets[0].getInstances();
			tune = subSets[1].getInstances();
			
			Trace.log("");
			Trace.log("Data Set Valid    ?", dataSet.verifyInstances());
			Trace.log("Feature Length    :", dataSet.getInstances()[0].feature.length);
			Trace.log("");
			Trace.log("Training Set Valid?", subSets[1].verifyInstances());
			Trace.log("Training Size     :", train.length);
			Trace.log("");
			Trace.log("Tuning Set Valid  ?", subSets[0].verifyInstances());
			Trace.log("Tuning Size       :", tune.length);

			long endTime = System.currentTimeMillis();
			long elapsedTime = endTime - startTime;
			Trace.log("");			
			Trace.log("Loading Time      :", elapsedTime / (1000) + "." + (elapsedTime % 1000)+"s");
		}
		catch (Exception ex) {
			ex.printStackTrace();
		}
	}
}