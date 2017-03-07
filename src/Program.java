
import java.util.ArrayList;
import java.util.List;
import ml.data.Amino;
import ml.data.DataSet;
import ml.data.Instance;
import ml.io.DataReader;
import ml.learner.neuralnet.NeuralNet;
import ml.learner.neuralnet.functions.*;
import ml.learner.neuralnet.initializers.*;
import ml.utils.tracing.*;
import ml.utils.Format;


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

		net.addLayer(2)
				.activationFunction(sigmoid)
				.weightInitializer(weightInit);

		net.addLayer(2)
				.activationFunction(sigmoid)
				.weightInitializer(weightInit);

		net.addLayer(2)
				.activationFunction(sigmoid)
				.weightInitializer(weightInit);

		double[][][] w = new double[3][][];
		w[0] = new double[2][2];
		w[1] = new double[2][3];
		w[2] = new double[2][3];

		w[1][0][0] = .35;
		w[1][0][1] = .15;
		w[1][0][2] = .20;

		w[1][1][0] = .35;
		w[1][1][1] = .25;
		w[1][1][2] = .30;

		w[2][0][0] = .60;
		w[2][0][1] = .40;
		w[2][0][2] = .45;

		w[2][1][0] = .60;
		w[2][1][1] = .50;
		w[2][1][2] = .55;

		Instance[] o = new Instance[1];
		o[0] = new Instance();
		o[0].features = new double[] { .05, .10 };
		o[0].target = new double[] { .01, .99 };

		net.weights(w);
		net.train(o, 0.5, 0, 0, 0.05, 100000);

		double[] f = new double[] { 0.05, 0.1 };
		double[] out = net.predict(f);
		Trace.log("[", Format.matrix(f), "]=[", Format.matrix(out), "]");
	}


	private static void sineExamples(int epochs) {
		double delta = Math.PI / 1000;
		double r = 0;
		List<Instance> m = new ArrayList<Instance>();

		while (r < (2 * Math.PI)) {
			Trace.log("sine(", r, ")=", Math.sin(r));
			r = r + delta;
			Instance x = new Instance();
			x.features = new double[] { r };
			x.target = new double[] { Math.sin(r) };
			m.add(x);
		}

		NeuralNet net = new NeuralNet();
		Function htan = new HyperbolicTangent();
		WeightInitializer weightInit = new DefaultWeightInitializer();
		Instance[] examples = m.toArray(new Instance[m.size()]);

		net.addLayer(1)
				.activationFunction(htan)
				.weightInitializer(weightInit);

		net.addLayer(100)
				.activationFunction(htan)
				.weightInitializer(weightInit);

		net.addLayer(100)
				.activationFunction(htan)
				.weightInitializer(weightInit);

		net.addLayer(100)
				.activationFunction(htan)
				.weightInitializer(weightInit);

		net.addLayer(1)
				.activationFunction(htan)
				.weightInitializer(weightInit);

		net.train(examples, 0.0025, 500, 0.00000005, 0.85, 0.00008);

		for (Instance t : m) {
			double[] out = net.predict(t.features);
			Trace.log("(", Format.matrix(t.target, 4), ",", Format.matrix(out, 4), ")");
		}

	}


	private static void xorExamples(int epochs) {

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

		net.addLayer(2)
				.activationFunction(sigmoid)
				.weightInitializer(weightInit);

		net.addLayer(4)
				.activationFunction(sigmoid)
				.weightInitializer(weightInit);

		net.addLayer(1)
				.activationFunction(sigmoid)
				.weightInitializer(weightInit);

		net.train(m, 0.5, 3000, 0.000000005, 0.7, 0.00008);

		for (Instance t : m) {
			double[] out = net.predict(t.features);
			Trace.log("[", Format.matrix(t.features, 4), "]=[", Format.matrix(out, 4), "]");
		}
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


	private static void proteinSecondary(String trainFile, String testFile, int[] hiddenUnits) {
		DataReader reader = null;

		try {

			StopWatch watch = StopWatch.start();

			reader = new DataReader(trainFile);
			List<List<Amino>> val = reader.Read();

			DataSet dataSet = new DataSet(val, 17);
			DataSet[] subSets = dataSet.Split(0.8);

			Instance[] train = subSets[0].instances();
			Instance[] tune = subSets[1].instances();

			reader = new DataReader(testFile);
			val = reader.Read();
			DataSet testSet = new DataSet(val, 17);
			Instance[] test = testSet.instances();

			Trace.log("");
			Trace.log("Data set valid    :", dataSet.verifyInstances());
			Trace.log("Feature length    :", dataSet.instances()[0].features.length);
			Trace.log("");
			Trace.log("Training set valid:", subSets[0].verifyInstances());
			Trace.log("Training size     :", train.length);
			Trace.log("");
			Trace.log("Tuning set valid  :", subSets[1].verifyInstances());
			Trace.log("Tuning size       :", tune.length);
			Trace.log("");
			Trace.log("Test set valid  :", testSet.verifyInstances());
			Trace.log("Test size       :", test.length);

			watch.stop();
			Trace.log("");
			Trace.log("Loading time      :", watch.elapsedTime() + "s");

			Function sigmoid = new Sigmoid();
			Function linear = new Linear();
			WeightInitializer weightInit = new DefaultWeightInitializer();

			int inputs = train[0].features.length;
			int outputs = train[0].target.length;

			for (int hu : hiddenUnits) {

				NeuralNet net = new NeuralNet();

				net.tune(tune);

				net.addLayer(inputs)
						.activationFunction(linear)
						.weightInitializer(weightInit);

				net.addLayer(hu)
						.activationFunction(linear)
						.weightInitializer(weightInit);

				net.addLayer(hu)
						.activationFunction(linear)
						.weightInitializer(weightInit);

				net.addLayer(outputs)
						.activationFunction(sigmoid)
						.weightInitializer(weightInit);

				watch = StopWatch.start();

				net.train(train, tune, 0.03, 100, 0.006, 0.75, 0.00005);

				watch.stop();

				int correct = 0;
				for (Instance t : test) {
					double[] out = net.predict(t.features);

					out = threshold(out);
					// Trace.log("([", Format.matrix(t.target, 0), "],[", Format.matrix(out, 0), "]");

					boolean match = true;
					for (int i = 0; i < t.target.length; i++) {
						if (t.target[i] != out[i]) {
							match = false;
						}
					}
					if (match) {
						correct++;
					}

					// Trace.log("([", Format.matrix(threshold(t.target), 0),"],[",Format.matrix(t.target,0), "])");
				}

				double acc = ((double) correct) / test.length;
				Trace.enabled = true;
				Trace.log(hu, ", ", Format.sprintf("%2.4f", acc), ", ", watch.elapsedTime(), ", ", net.epoch());
			}
		}
		catch (Exception ex) {
			ex.printStackTrace();
		}

	}


	public static void main(String[] args) {

		int[] hu = new int[] { 1, 2, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110,
				120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220,
				230, 240, 250, 300, 400, 500, 600, 700, 800, 900, 1000 };

		proteinSecondary(args[0], args[1], hu);
		// stepByStepExamples();
		// xorExamples(3000);

		// sineExamples(10000);
	}

}