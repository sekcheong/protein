import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;

/***************************************************
 Assignment 3   - Spring 2004
 Original Code Copied from: Mikhail Afanasyev
 Course: CS 182
 
 Assignment 3: Backpropagation
 
 Modified by Liam Mac Dermed for gradehelper
 **************************************************/

/**
 * Class for testing backpropagation networks.
 */
public class TesterPart2 {
	
	/**
	 * Create a network for learning the AND function.
	 */
	static public Net createAnd () {
	  int[][][] data = loadTrainingDataFromFile("and.data");
		return new Net(data[0][0].length, 1, 1, data[1][0].length, data[0], data[1]);
	}
	
	// network for leaning OR function
	static public Net createOr () {
    int[][][] data = loadTrainingDataFromFile("or.data");
		return new Net(data[0][0].length, 1, 1, data[1][0].length, data[0], data[1]);
	}
	
	/**
	 * Create a network for learning the SAME function.
	 */
	static public Net createSame (){
    int[][][] data = loadTrainingDataFromFile("same.data");
		return new Net(data[0][0].length, 1, 2, data[1][0].length, data[0], data[1]);
	}
	
	/**
	 * Create network for leaning N-input, M-hidden node auto encoder
	 */
	
	static public Net createAutoCoder(int inputs, int hidden) {
		int patterns[][]  = new int[inputs] [inputs];
		for (int i=0; i<inputs; i++) {
			patterns[i] = new int[inputs];
			for (int j=0; j<inputs; j++)
				patterns[i][j] = (i==j)?1:0;
		};
		return new Net(inputs, 1, hidden, inputs, patterns, patterns);
	};
	
	/**
	 * The function should be called as:
	 *    % java TesterPart2 datafile ne lr mom ec hidLayers nhid
	 * where:
	 *    datafile contains the data to be learned
	 *    ne is the number of epochs
	 *    lr is the learning rate
	 *    mom is the momentum
	 *    ec is the error criterion
   *    hidLayers is the number of hidden layers
   *    nhid is the number of neurons per hidden layer
	 */
	public static void main(String[] argv) {
		if (argv.length != 7 && argv.length != 0) {
			System.err.println("Invalid argument count");
			System.err.println("The function should be called as:");
			System.err.println("    % java TesterPart2");
      System.err.println(" in order to run automated tests; or as");
      System.err.println("    % java TesterPart2 datafile ne lr mom ec hidLayers nhid");
      System.err.println(" where:");
      System.err.println("    datafile contains the data to be learned");
      System.err.println("    ne is the number of epochs");
      System.err.println("    lr is the learning rate");
      System.err.println("    mom is the momentum");
      System.err.println("    ec is the error criterion");
      System.err.println("    hidLayers is the number of hidden layers");
      System.err.println("    nhid is the number of neurons per hidden layer");
      return;
    }
		if(argv.length == 0) {
			Net n;
			
			System.out.print("\n   Attempting to train AND:        \n");
			
			n = createAnd();
			n.train();
			n.logNetwork();
			
			System.out.print("\n   Attempting to train SAME:        \n");
			n = createSame();
			n.train();
			n.logNetwork();
			
			System.out.print("\n   Attempting to train a  4-2-4 autoencoder:        \n");
			n = createAutoCoder(4, 2);
			n.train();
      n.logNetwork();
			
			
		} else {
			
      int[][][] data = loadTrainingDataFromFile(argv[0]);
      int hidLayers = Integer.parseInt(argv[5]);
      int nhid = Integer.parseInt(argv[6]);
      Net n = new Net(data[0][0].length, hidLayers, nhid, data[1][0].length, data[0], data[1]);
			
			n.setTrainingParameters(Integer.parseInt(argv[1]),
					Double.parseDouble(argv[2]),
					Double.parseDouble(argv[3]),
					Double.parseDouble(argv[4]));
			
			n.train();
    }
	}
  
  /**
   * Load training data out of a file.
   * 
   * @param filename
   */
  public static int[][][] loadTrainingDataFromFile(String filename) {
    int[][] trainingData = null;
    int[][] trainingTargets = null;
    try {
      BufferedReader in = new BufferedReader(new FileReader(filename));
      String s;
      int state = 0, count=0;
      int nData = 0,
        nIn = 0,
        nOut = 0;
      while ((s=in.readLine())!=null) {
        String[] parts = s.split(" ");
        switch(state) {
        case 0:
          if (parts[0].equals("DATA_DESCRIPTION"))
            state=1;
          break;
        case 1:
          nData = Integer.parseInt(parts[0]);
          nIn = Integer.parseInt(parts[1]);
          nOut = Integer.parseInt(parts[2]);
          trainingData = new int[nData][nIn];
          trainingTargets = new int[nData][nOut];
          state=2;
          break;
        case 2:
          if (parts[0].equals("DATA"))
            state=3;
          break;
        case 3:
          if (parts.length!=nIn+nOut+1)
            throw new Exception("failed reading file");
          trainingData[count] = parseNInts(parts,0,nIn);
          trainingTargets[count] = parseNInts(parts,nIn+1,nOut);
          count++;
          break;
        }
      }
    } catch (Exception e) {
      throw new Error("failed reading file "+filename);
    }
    int[][][] result = {trainingData,trainingTargets};
    return result;
  }
 
  
  /**
   *  Parse an array of strings into an array of doubles.
   * 
   * @param arr
   * @return
   */
  public static double[] parseDoubleArray(String[] arr) {
    ArrayList<String> pruned = new ArrayList<String>();
    for (int i=0; i<arr.length; i++) {
      if (!arr[i].equals(""))
        pruned.add(arr[i]);
    }
    arr = pruned.toArray(new String[0]); 
    double[] d = new double[arr.length];
    for (int i=0; i<arr.length; i++) {
      d[i] = Double.parseDouble(arr[i]);
    }
    return d;
  }
  
  /** Parse part of an array of integers.
   * 
   * @param arr The array to parse
   * @param start The element in the array to start with
   * @param count The number of elements to parse
   * @return the parsed integers
   */
  public static int[] parseNInts(String[] arr, int start, int count) {
    int[] d = new int[count];
    for (int i=0; i<count; i++) {
      d[i] = Integer.parseInt(arr[i+start]);
    }
    return d;
  }

  /**
   * Log the momentum of an array of units.
   */
  public static void logUnitArrayMomentum(Unit[] units) {
    for (int i=0; i<units.length; i++) {
      System.out.print(""+Net.arrayToString(units[i].weightChangeMomentum)+" ");
      if (i!=units.length-1)
        System.out.print("; ");
    }
    System.out.println();
  }
  
}
