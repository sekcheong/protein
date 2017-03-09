
import java.util.List;
import ml.data.Amino;
import ml.data.DataSet;
import ml.data.Instance;
import ml.io.DataReader;
import ml.learner.neuralnet.NeuralNet;
import ml.learner.neuralnet.functions.*;
import ml.learner.neuralnet.initializers.*;
import ml.utils.tracing.*;
import ml.utils.Console;
import ml.utils.Format;

public class Tests {

	private static double[] threshold(double[] values) {
		double[] t = new double[values.length];
		int max = 0;
		for (int i = 0; i < values.length; i++) {
			if (values[i] > values[max]) {
				max = i;
			}
		}
		t[max] = 1;
		return t;
	}


	private static double checkAccuracy(NeuralNet net, Instance[] test) {
		int correct = 0;
		for (Instance t : test) {
			double[] out = net.predict(t.features);

			out = threshold(out);

			boolean match = true;
			for (int i = 0; i < t.target.length; i++) {
				if (t.target[i] != out[i]) {
					match = false;
				}
			}
			if (match) {
				correct++;
			}
		}
		double acc = ((double) correct) / test.length;
		return acc;
	}


	private static double runTest(String inputFile, int hiddenUnits, int maxEpoch, double eta, double epsilon, double alpha, double lambda) {
		DataReader reader = null;
		Instance[] train = null;
		Instance[] tune = null;
		Instance[] test = null;

		try {

			reader = new DataReader(inputFile);
			List<List<Amino>> val = reader.Read();

			DataSet dataSet = new DataSet(val, 17);

			DataSet[] sets = dataSet.Split();

			train = sets[0].instances();
			tune = sets[1].instances();
			test = sets[2].instances();
		}
		catch (Exception ex) {
			ex.printStackTrace();
		}

		if (train.length < 3000) {
			maxEpoch = 100;
		}

		Function sigmoid = new Sigmoid();
		Function linear = new Linear();

		WeightInitializer weightInit = new DefaultWeightInitializer();

		int inputs = train[0].features.length;
		int outputs = train[0].target.length;

		NeuralNet net = new NeuralNet();

		net.addLayer(inputs)
				.activationFunction(linear)
				.weightInitializer(weightInit);

		net.addLayer(hiddenUnits)
				.activationFunction(linear)
				.weightInitializer(weightInit);

		net.addLayer(outputs)
				.activationFunction(sigmoid)
				.weightInitializer(weightInit);

		net.train(train, tune, eta, maxEpoch, epsilon, alpha, lambda);

		double accuracy = checkAccuracy(net, test);

		return accuracy;
	}


	// test without momentum and weight decay
	private static void experiment(String inputFile, int hiddenUnits, int maxEpoch, double eta, double epsilon, double alpha, double lambda, int trails) {

		Console.writeLine("% hiddenUnits:", hiddenUnits);
		Console.writeLine("% maxEpoch   :", maxEpoch);
		Console.writeLine("% eta        :", eta);
		Console.writeLine("% epsilon    :", epsilon);
		Console.writeLine("% alpha      :", alpha);
		Console.writeLine("% lambda     :", lambda);

		double[] time = new double[trails];
		double[] accu = new double[trails];
		double min = 99999;
		double max = -1;
		double avg = 0;
		double avgTime = 0;

		for (int i = 0; i < trails; i++) {
			StopWatch watch = StopWatch.start();
			double acc = runTest(inputFile, hiddenUnits, maxEpoch, eta, epsilon, alpha, lambda);
			watch.stop();
			time[i] = watch.elapsedTime();
			accu[i] = acc;
			if (acc > max) max = acc;
			if (acc < min) min = acc;
		}

		avg = ml.math.Vector.sigma(accu) / accu.length;
		avgTime = ml.math.Vector.sigma(time) / time.length;
		Console.writeLine("");
		Console.writeLine(avg * 100, ", ", min * 100, ", ", max * 100, ",", avgTime, ",", hiddenUnits, ",", eta, ",", epsilon, ",", alpha, ",", lambda);
	}


	public static void main(String[] args) {
		if (args.length < 1) {
			Console.writeLine("Usage:");
			Console.writeLine("  Tests [input] [trails]");
			return;
		}

		int trails = 3;

		if (args.length > 1) {
			try {
				trails = Integer.parseInt(args[1]);
			}
			catch (Exception ex) {
				Console.writeLine("Error:" + ex.getMessage());
			}
		}

		Console.writeLine("%% Number of trails ", trails);
		Console.writeLine("%% Test without regularizations");

		int hiddenUnits = 9;
		int maxEpoch = 100;

		double eta = 0.01;
		double epsilon = 0.001;
		double alpha = 0.0;
		double lambda = 0.0;

		experiment(args[0], hiddenUnits, maxEpoch, eta, epsilon, alpha, lambda, trails);
		Console.writeLine("");

		eta = 0.01;
		epsilon = 0.001;
		alpha = 0.75;
		lambda = 0.0;

		Console.writeLine("%% Test hidden units");
		for (int hu : new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 40, 50, 70, 90, 100, 110, 130, 150, 160, 170, 180, 190, 200, 250, 300, 350, 400, 450, 500 }) {
			experiment(args[0], hu, maxEpoch, eta, epsilon, alpha, lambda, trails);
			Console.writeLine("");
		}

		
		eta = 0.01;
		epsilon = 0.001;
		alpha = 0.0;
		lambda = 0.0;

		Console.writeLine("%% Test momentum terms (alpha)");
		for (double a : new double[] { 0.05, 0.10, 0.15, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.75, 0.80, 0.85, 0.90, 0.95 }) {
			experiment(args[0], hiddenUnits, maxEpoch, eta, epsilon, a, lambda, trails);
			Console.writeLine("");
		}

		Console.writeLine("%% Test weight decay (lambda)");
		for (double l : new double[] { 0.000003, 0.000005, 0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006, 0.00007, 0.00008, 0.00009, 0.001 }) {
			experiment(args[0], hiddenUnits, maxEpoch, eta, epsilon, alpha, l, trails);
			Console.writeLine("");
		}

		
		maxEpoch = 250;
		eta = 0.01;
		epsilon = 0.001;
		alpha = 0.75;
		lambda = 0.0;
		Console.writeLine("%% Test learning rate (eta)");
		for (double e : new double[] { 0.0005, 0.0007, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09 }) {
			experiment(args[0], hiddenUnits, maxEpoch, e, epsilon, alpha, lambda, trails);
			Console.writeLine("");
		}
	}

}