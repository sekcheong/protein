
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


	private static Instance[] makeSimpleExamples() {
		Instance[] examples = new Instance[1];
		Instance x;
		x = new Instance();
		x.features = new double[] { 0.3, 0.7 };
		x.target = new double[] { 1 };
		examples[0] = x;
		return examples;
	}


	private static void stepByStepExamples() {
		NeuralNet net = new NeuralNet();

		Function sigmoid = new Sigmoid();
		WeightInitializer weightInit = new DefaultWeightInitializer();

		net.addLayer(2).activationFunction(sigmoid).weightInitializer(weightInit);
		net.addLayer(2).activationFunction(sigmoid).weightInitializer(weightInit);
		net.addLayer(2).activationFunction(sigmoid).weightInitializer(weightInit);

		net._w = new double[3][][];
		net._w[0] = new double[2][2];
		net._w[1] = new double[2][3];
		net._w[2] = new double[2][3];

		net._w[1][0][0] = .35;
		net._w[1][0][1] = .15;
		net._w[1][0][2] = .20;

		net._w[1][1][0] = .35;
		net._w[1][1][1] = .25;
		net._w[1][1][2] = .30;

		net._w[2][0][0] = .60;
		net._w[2][0][1] = .40;
		net._w[2][0][2] = .45;

		net._w[2][1][0] = .60;
		net._w[2][1][1] = .50;
		net._w[2][1][2] = .55;

		Instance[] o = new Instance[1];
		o[0] = new Instance();
		o[0].features = new double[] { .05, .10 };
		o[0].target = new double[] { .01, .99 };

		net.train(o, 0.5, 0, 0, 0.05, 100000);
		double[] f = new double[] { 0.05, 0.1 };
		double[] out = net.predict(f);
		Trace.log("[", Format.matrix(f), "]=[", Format.matrix(out), "]");
	}


	private static void xorExamples() {

		Instance[] m = new Instance[4];

		m[0] = new Instance();
		m[0].features = new double[] { 0, 0 };
		m[0].target = new double[] { 0 };

		m[1] = new Instance();
		m[1].features = new double[] { 0, 1 };
		m[1].target = new double[] { 1 };

		m[2] = new Instance();
		m[2].features = new double[] { 1, 0 };
		m[2].target = new double[] { 1 };

		m[3] = new Instance();
		m[3].features = new double[] { 1, 1 };
		m[3].target = new double[] { 0 };

		NeuralNet net = new NeuralNet();

		Function sigmoid = new Sigmoid();
		WeightInitializer weightInit = new DefaultWeightInitializer();

		net.addLayer(2).activationFunction(sigmoid).weightInitializer(weightInit);
		net.addLayer(4).activationFunction(sigmoid).weightInitializer(weightInit);
		net.addLayer(4).activationFunction(sigmoid).weightInitializer(weightInit);
		net.addLayer(1).activationFunction(sigmoid).weightInitializer(weightInit);

		net.train(m, 0.5, 0, 0, 0.005, 10000);

		for (Instance t : m) {
			double[] out = net.predict(t.features);
			Trace.log("[", Format.matrix(t.features), "]=[", Format.matrix(out), "]");
		}
	}


	private static void proteinSecondary(String trainFile, String testFile) {
		DataReader reader = null;
		Instance[] train = null;
		Instance[] tune = null;

		try {

			StopWatch watch = StopWatch.start();

			reader = new DataReader(trainFile);
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

			Function sigmoid = new Sigmoid();
			Function linear = new Linear();
			WeightInitializer weightInit = new DefaultWeightInitializer();

			int inputs = train[0].features.length;
			int outputs = train[0].target.length;

			NeuralNet net = new NeuralNet();

			net.addLayer(inputs)
					.activationFunction(linear)
					.weightInitializer(weightInit);

			net.addLayer(1000)
					.activationFunction(linear)
					.weightInitializer(weightInit);

			net.addLayer(400)
					.activationFunction(sigmoid)
					.weightInitializer(weightInit);
			
			net.addLayer(outputs)
					.activationFunction(sigmoid)
					.weightInitializer(weightInit);

			Trace.log("Learning...");
			watch = StopWatch.start();

			Trace.enabled=false;
			net.train(train, 0.5, 0, 0, 0.01, 30);
			Trace.enabled=true;
			
			watch.stop();
			Trace.log("[done]");
			Trace.log("Elpased time      :", watch.elapsedTime() + "s");
			
			for (Instance t: tune) {
					double[] out = net.predict(t.features);
					double max = -1;
					int maxIndex = 0;
					
					for (int i=0; i<out.length; i++) {
						if (out[i]>0.5) { 
							out[i]=1;
//							maxIndex =i;
//							max = out[i];
						}
//						out[i]=0;
					}
					
					out[maxIndex]=1;
					Trace.log("([", Format.matrix(t.target), "],[", Format.matrix(out), "])");
			}

		}
		catch (Exception ex) {
			ex.printStackTrace();
		}

	}


	public static void main(String[] args) {
		Trace.enabled = true;

		proteinSecondary(args[0], args[1]);
		
		// stepByStepExamples();

		//xorExamples();

		// use debug examples
		// train = makeExamples();

		// use default weight initializer
		// WeightInitializer weightInit = new DefaultWeightInitializer();
		//
		// Function linear = new Linear();
		// Function ligistic = new Sigmoid();
		// Function htan = new HyperbolicTangent();

		// NeuralNet neuralNet = new NeuralNet();
		//
		// train = makeExamples(); // makeXorExamples(); // makeXorExamples();
		//
		// int inputs = train[0].features.length;
		// int outputs = train[0].target.length;
		//
		// neuralNet.addLayer(inputs);
		//
		// neuralNet.addLayer(2)
		// .weightInitializer(weightInit)
		// .activationFunction(ligistic);
		//
		// neuralNet.addLayer(outputs)
		// .weightInitializer(weightInit)
		// .activationFunction(ligistic);
		//
		// neuralNet.train(train, 0.005, 0, 0, 0.01, 100);

		//Trace.log("[END]");

		// double[] ans;
		//
		// ans = neuralNet.predict(new double[] {0,0});
		// Trace.log(Format.matrix(ans));
		//
		// ans = neuralNet.predict(new double[] {0,1});
		// Trace.log(Format.matrix(ans));
		//
		// ans = neuralNet.predict(new double[] {1,0});
		//
		// Trace.log(Format.matrix(ans));
		// ans = neuralNet.predict(new double[] {1,1});
		//

		// for (Instance ex : tune) {
		// double[] y = neuralNet.predict(ex.features);
		// Trace.log("y=[", Format.matrix(y), "]");
		// Trace.log("t=[", Format.matrix(ex.target), "]");
		// }
		// double[] ans = neuralNet.predict(new double[] { 0.2, 0.3 });
		// Trace.log("y_hat=[", Format.matrix(ans), "]");
	}

}