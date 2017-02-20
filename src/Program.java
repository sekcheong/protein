
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
//			List<Instance> insts = subSets.get(0).getInstances();
//			insts = subSets.get(1).getInstances();
		}
		catch (Exception ex) {
			ex.printStackTrace();
		}		
	}
}