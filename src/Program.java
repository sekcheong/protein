
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


public class Program {

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


	private static double proteinSecondary(String dataFile, int hiddenUnits, double eta, int maxEpoch, double epsilon, double alpha, double lambda) {
		DataReader reader = null;

		try {

			StopWatch watch = StopWatch.start();

			reader = new DataReader(dataFile);
			List<List<Amino>> val = reader.Read();

			DataSet dataSet = new DataSet(val, 17);

			DataSet[] sets = dataSet.Split();

			Instance[] train = sets[0].instances();
			Instance[] tune = sets[1].instances();
			Instance[] test = sets[2].instances();

			watch.stop();
			// Trace.log("");
			// Trace.log("Loading time :", watch.elapsedTime() + "s");

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

//			Trace.log("");
//			Trace.log("Hidden units: ", hiddenUnits);
//			Trace.log("Max epoch   : ", maxEpoch);
//			Trace.log("eta         : ", Format.sprintf("%1.8f", eta));
//			Trace.log("epsilon     : ", Format.sprintf("%1.8f", epsilon));
//			Trace.log("alpha       : ", Format.sprintf("%1.8f", alpha));
//			Trace.log("lambda      : ", Format.sprintf("%1.8f", lambda));

			watch = StopWatch.start();

			net.train(train, tune, eta, maxEpoch, epsilon, alpha, lambda);

			watch.stop();

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

//			Trace.log("");
//			Trace.log(Format.sprintf("Accuracy    : %2.4f", acc));
//			Trace.log("Epoch       : ", net.epoch());
//			Trace.log("Elapsed time: ", watch.elapsedTime());
//			Trace.log("");
			return acc;
		}
		catch (Exception ex) {
			ex.printStackTrace();
		}
		return 0;

	}


	public static void main(String[] args) {

		int hiddenUnits = 1;

		double eta = 0.01;

		int maxEpoch = 1000;

		double epsilon = 0.009;

		double alpha = 0.75;

		double lambda = 0.000008;

		String dataFile = args[0];

		// try {
		// dataFile = args[0];
		//
		// if (args.length > 2) {
		// hiddenUnits = Integer.parseInt(args[2]);
		// }
		//
		// if (args.length > 3) {
		// eta = Double.parseDouble(args[3]);
		// }
		//
		// if (args.length > 4) {
		// maxEpoch = Integer.parseInt(args[4]);
		// }
		//
		// if (args.length > 5) {
		// epsilon = Double.parseDouble(args[5]);
		// }
		//
		// if (args.length > 6) {
		// alpha = Double.parseDouble(args[6]);
		// }
		//
		// if (args.length > 7) {
		// lambda = Double.parseDouble(args[7]);
		// }
		// }
		// catch (Exception ex) {
		// Console.writeLine("Invalid parameters.");
		// }

		for (int i=1; i<=100; i=i+3) {
			double[] acc = new double[10];
			for (int j=0; j<10; j++) {
				acc [j]= proteinSecondary(dataFile, i, eta, maxEpoch, epsilon, alpha, lambda);
			}
			double avg = ml.math.Vector.sigma(acc) / acc.length;
			Console.writeLine(i + "," + avg);
		}

	}

}