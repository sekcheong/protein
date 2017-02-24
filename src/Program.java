
import java.util.List;
import ml.data.Amino;
import ml.data.DataSet;
import ml.data.Instance;
import ml.io.DataReader;
import ml.learner.nerualnet.Layer;
import ml.learner.nerualnet.Net;
import ml.learner.nerualnet.functions.Linear;
import ml.learner.nerualnet.functions.Sigmoid;
import ml.learner.neuralnet.initializers.DefaultWeightInitializer;
import ml.learner.neuralnet.initializers.WeightInitializer;
import ml.utils.tracing.StopWatch;
import ml.utils.tracing.Trace;

public class Program {

	private static Instance[] makeExamples() {
		Instance[] examples = new Instance[4];
		Instance x;
		x = new Instance(3, 2);
		x.features[0] = 0.2;
		x.features[1] = 0.9;
		x.features[2] = 0.4;
		x.target[0] = 0.7;
		x.target[1] = 0.3;
		examples[0] = x;

		x = new Instance(3, 2);
		x.features[0] = 0.1;
		x.features[1] = 0.3;
		x.features[2] = 0.5;
		x.target[0] = 0.6;
		x.target[1] = 0.4;
		examples[1] = x;

		x = new Instance(3, 2);
		x.features[0] = 0.9;
		x.features[1] = 0.7;
		x.features[2] = 0.8;
		x.target[0] = 0.9;
		x.target[1] = 0.5;
		examples[2] = x;

		x = new Instance(3, 2);
		x.features[0] = 0.6;
		x.features[1] = 0.4;
		x.features[2] = 0.3;
		x.target[0] = 0.2;
		x.target[1] = 0.8;
		examples[3] = x;
		return examples;
	}

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
			Trace.log("Data set valid    :", dataSet.verifyInstances());
			Trace.log("Feature length    :", dataSet.instances()[0].features.length);
			Trace.log("");
			Trace.log("Training set valid:", subSets[1].verifyInstances());
			Trace.log("Training size     :", train.length);
			Trace.log("");
			Trace.log("Tuning set valid  :", subSets[0].verifyInstances());
			Trace.log("Tuning size       :", tune.length);

			watch.stop();

			Trace.log("");
			Trace.log("Loading time      :", watch.elapsedTime() + "s");
		}
		catch (Exception ex) {
			ex.printStackTrace();
		}

		// use debug examples
		train = makeExamples();

		// use default weight initializer
		WeightInitializer weightInit = new DefaultWeightInitializer();

		// create a neural network with one input layer, one hidden
		// layer, and output layer
		Net neuralNet = new Net();

		
		Layer input = new Layer(4);
		
		Layer hidden1 = new Layer(4, 3);
		hidden1.weightInitializer(weightInit);
		hidden1.activationFunction(new Sigmoid());
		
		Layer hidden2 = new Layer(3, 2);
		hidden2.weightInitializer(weightInit);
		hidden2.activationFunction(new Sigmoid());
		 
		Layer output = new Layer(2, 1);
		output.weightInitializer(weightInit);
		output.activationFunction(new Linear());

		neuralNet.addLayer(input);
		neuralNet.addLayer(hidden1);
		neuralNet.addLayer(hidden2);
		neuralNet.addLayer(output);

		neuralNet.train(train, 0.05, 0.4, 5);

	}

}