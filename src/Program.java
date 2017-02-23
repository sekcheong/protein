
import java.util.List;
import ml.data.Amino;
import ml.data.DataSet;
import ml.data.Instance;
import ml.io.DataReader;
//import ml.utils.DataShuffler;
import ml.utils.tracing.StopWatch;
import ml.utils.tracing.Trace;

public class Program {

	public static void main(String[] args) {
		Trace.enabled = true;

		DataReader reader = null;
		Instance[] train = null;
		Instance[] tune = null;		

		try {
			
			StopWatch watch = StopWatch.start();
						
			reader = new DataReader(args[0]);			
			List<List<Amino>> val = reader.Read();
			
			DataSet dataSet = new DataSet(val, 17);			
			DataSet[] subSets = dataSet.Split(0.2);
			train = subSets[0].instances();
			tune = subSets[1].instances();
			
			Trace.log("");
			Trace.log("Data Set Valid    ?", dataSet.verifyInstances());
			Trace.log("Feature Length    :", dataSet.instances()[0].feature.length);
			Trace.log("");
			Trace.log("Training Set Valid?", subSets[1].verifyInstances());
			Trace.log("Training Size     :", train.length);
			Trace.log("");
			Trace.log("Tuning Set Valid  ?", subSets[0].verifyInstances());
			Trace.log("Tuning Size       :", tune.length);

			watch.stop();

			Trace.log("");
			Trace.log("Loading Time      :", watch.elapsedTime() + "s");
		}
		catch (Exception ex) {
			ex.printStackTrace();
		}
	}
}