import java.util.List;
import java.util.Vector;

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

public class Lab2 {
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


	private static void printResults(NeuralNet net, Instance[] test) {
		int correct = 0;
		Console.writeLine("Actual  Predicted");
		Console.writeLine("=================");
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

			Console.writeLine(getStructureName(t.target), "     ", getStructureName(out));
		}
		double acc = ((double) correct) / test.length;
		Console.writeLine("");
		Console.writeLine("Total samples      :", test.length);
		Console.writeLine("Corrected predicted:", correct);
		Console.writeLine("Accuracy           :", Format.sprintf("%1.4f", acc * 100), "%");
	}


	private static String getStructureName(double[] v) {
		if (v[0] > 0) return "_";
		if (v[1] > 0) return "h";
		if (v[2] > 0) return "e";
		return "";
	}


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


	private static void predictSecondaryProtein(String dataFile) {

		int hiddenUnits = 9;

		double eta = 0.05;

		int maxEpoch = 50;

		double epsilon = 0.007;

		double alpha = 0.725;

		double lambda = 0.000008;

		DataReader reader = null;
		Instance[] train = null;
		Instance[] tune = null;
		Instance[] test = null;

		try {

			reader = new DataReader(dataFile);
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

		double[][][][] savedWeights;
		double maxAccuracy = 0;
		int max = 0;
		int models = 5;

		savedWeights = new double[models][][][];
		double[] accuracys = new double[models];

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

		// train x models and take the most accurate one
		for (int i = 0; i < models; i++) {

			net.train(train, tune, eta, maxEpoch, epsilon, alpha, lambda);

			savedWeights[i] = net.weights();

			double accuracy = checkAccuracy(net, test);

			if (accuracy > maxAccuracy) {
				max = i;
				maxAccuracy = accuracy;
			}
		}

		net.weights(savedWeights[max]);
		printResults(net, test);

	}


	public static void main(String[] args) {
		if (args.length < 1) {
			Console.writeLine("Usage:");
			Console.writeLine("  lab2 [input]");
			return;
		}
		predictSecondaryProtein(args[0]);
	}

}
