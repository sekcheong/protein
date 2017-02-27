
import java.util.List;
import ml.data.Amino;
import ml.data.DataSet;
import ml.data.Instance;
import ml.io.DataReader;
import ml.learner.neuralnet.NeuralNet;
import ml.learner.neuralnet.functions.*;
import ml.learner.neuralnet.initializers.*;
import ml.utils.Format;
import ml.utils.tracing.*;

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

		Function hiddenFunc = new Linear();
		Function outputFunc = new Sigmoid();
		Function htan = new HyperbolicTangent();

		NeuralNet neuralNet = new NeuralNet();

		// 2 inputs
		neuralNet.addLayer(2);

		// 3 units
		neuralNet.addLayer(3).weightInitializer(weightInit).activationFunction(htan);

		// 2 units
		neuralNet.addLayer(2).weightInitializer(weightInit).activationFunction(htan);

		// // 2 units
		// neuralNet.addLayer(2)
		// .weightInitializer(weightInit)
		// .activationFunction(htan);
		//
		// // 2 units
		// neuralNet.addLayer(2)
		// .weightInitializer(weightInit)
		// .activationFunction(htan);
		//
		// 1 output unit
		neuralNet.addLayer(2).weightInitializer(weightInit).activationFunction(htan);

		neuralNet.train(train, 0.05, 0, 0.04, 20);

		double[] ans = neuralNet.predict(new double[] { 0.2, 0.3 });
		Trace.log("y_hat=[", Format.matrix(ans), "]");
	}

}