
import java.util.List;
import ml.data.Amino;
import ml.data.DataSet;
import ml.data.Instance;
import ml.io.DataReader;
import ml.learner.nerualnet.Layer;
import ml.learner.nerualnet.Net;
import ml.learner.nerualnet.config.DefaultWeightInitializer;
import ml.learner.nerualnet.config.WeightInitializer;
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
			Trace.log("Data Set Valid    :", dataSet.verifyInstances());
			Trace.log("Feature Length    :", dataSet.instances()[0].feature.length);
			Trace.log("");
			Trace.log("Training Set Valid:", subSets[1].verifyInstances());
			Trace.log("Training Size     :", train.length);
			Trace.log("");
			Trace.log("Tuning Set Valid  :", subSets[0].verifyInstances());
			Trace.log("Tuning Size       :", tune.length);

			watch.stop();

			Trace.log("");
			Trace.log("Loading Time      :", watch.elapsedTime() + "s");
		}
		catch (Exception ex) {
			ex.printStackTrace();
		}

		// use default weight initializer
		WeightInitializer weightInit = new DefaultWeightInitializer();

		// create a neural network with one input layer, one hidden
		// layer, and output layer
		Net neuralNet = new Net();

		// create the input layer has 18 inputs including the bias
		Layer input = new Layer(18, 18);
		input.weightInitializer(weightInit);

		// create the hidden layer with 18 inputs and 21 units
		Layer hidden = new Layer(18, 21);
		hidden.weightInitializer(weightInit);
		
		// create the hidden layer with 18 inputs and 21 units
		Layer hidden2 = new Layer(21, 21);
		hidden2.weightInitializer(weightInit);

		// create the output layer with 21 inputs and 3 output units
		Layer output = new Layer(21, 3);
		output.weightInitializer(weightInit);

		neuralNet.addLayer(input);
		neuralNet.addLayer(hidden);
		neuralNet.addLayer(hidden2);
		neuralNet.addLayer(output);

		neuralNet.initialize();

	}

}