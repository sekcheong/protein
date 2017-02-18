//import java.io.BufferedReader;
//import java.io.FileReader;
//import java.util.Arrays;
//
///***************************************************
// Assignment 3-part1   - Spring 2005
// Course: CS 182
// CS182, Assignment 3: Backpropagation
// TesterPart1.java
// **************************************************/
//
///**
// * Class for testing backpropagation networks.
// */
//public class TesterPart1 {
//
//  static int[][] data = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
//
//  static int[][] andTargets = { { 0 }, { 0 }, { 0 }, { 1 } };
//
//  static int[][] orTargets = { { 0 }, { 1 }, { 1 }, { 1 } };
//
//  static int[][] sameTargets = { { 1 }, { 0 }, { 0 }, { 1 } };
//
//  /**
//   * Create a network for learning the AND function.
//   */
//  static public Net createAnd() {
//    return new Net(data, andTargets);
//  }
//
//  // network for leaning OR function
//  static public Net createOr() {
//    return new Net(data, orTargets);
//  }
//
//  /**
//   * Create a network for learning the SAME function.
//   */
//  // static public Net createSame (){
//  // return new Net(5, 2, 1, data, sameTargets);
//  // }
//  /**
//   * Create a network w/o hidden layers that won't learning the SAME function.
//   */
//  static public Net createBadSame() {
//    return new Net(data, sameTargets);
//  }
//
//  /**
//   * The function should be called as: 
//   *   % java TesterPart1
//   * (This runs some simple sanity checks.) or as:
//   *   % java TesterPart1 training_file ne lr mom ec, where:
//   * training_file is a file containing the function to be learned ("and.train", "or.train", etc...)
//   * ne is the number of epochs
//   * lr is the learning rate
//   * mom is the momentum
//   * ec is the error criterion
//   */
//
//  public static void main(String[] argv) {
//
//    if (argv.length == 0) {
//      System.out
//          .print("Initiating Simple Sanity Tests (does not guarantee correctness):\n");
//      // System.out.print("These Functions are in dependent order. Failing an
//      // earlier one, will lead to failing later ones, even if they are correcly
//      // written.");
//      System.out.print("Testing Unit.java Class:\n");
//      System.out.print("   initialize() Test:            ");
//
//      Unit Bias = new Unit();
//      Bias.activation = 1.0;
//
//      Unit inUnit1 = new Unit();
//      Unit inUnit2 = new Unit();
//      Unit outUnit = new Unit();
//      inUnit1.setOutgoingUnit(outUnit);
//      inUnit2.setOutgoingUnit(outUnit);
//      outUnit.addIncomingUnit(inUnit1);
//      outUnit.addIncomingUnit(inUnit2);
//      outUnit.addIncomingUnit(Bias);
//
//      outUnit.initialize();
//
//      if (Net.MIN_WEIGHT <= outUnit.inWeights[0]
//          && Net.MAX_WEIGHT >= outUnit.inWeights[0]
//          && Net.MIN_WEIGHT <= outUnit.inWeights[1]
//          && Net.MAX_WEIGHT >= outUnit.inWeights[1]
//          && Net.MIN_WEIGHT <= outUnit.inWeights[2]
//          && Net.MAX_WEIGHT >= outUnit.inWeights[2]) {
//        System.out.print("PASSED \n");
//      } else {
//        System.out.print("FAILED \n");
//      }
//
//      System.out.print("   computeActivation() Test:     ");
//
//      outUnit.inWeights[0] = 0.5;
//      outUnit.inWeights[1] = 0.5;
//      outUnit.inWeights[2] = 0.5;
//
//      inUnit1.activation = 1.0;
//      inUnit2.activation = 1.0;
//      outUnit.computeActivation();
//
//      if (outUnit.activation >= 0.8175 && outUnit.activation <= 0.8176) {// 1.5
//                                                                          // /
//                                                                          // (1.0
//                                                                          // +
//                                                                          // Math.exp(1.5)
//                                                                          // ) )
//                                                                          // {
//        System.out.print("PASSED \n");
//      } else {
//        System.out.print("FAILED with activation " + outUnit.activation
//            + " expecting activation 0.8175 \n");
//      }
//
//      System.out.print("   computeError() Test:          ");
//
//      outUnit.computeError(0);
//      if (outUnit.error >= -0.8176 && outUnit.error <= -0.8175) {// 1.5 / (1.0
//                                                                  // +
//                                                                  // Math.exp(1.5)
//                                                                  // ) ) {
//        System.out.print("PASSED \n");
//      } else {
//        System.out.print("FAILED with error " + outUnit.error
//            + " expecting error  -0.8175 \n");
//      }
//
//      System.out.print("   computeWeightChange() Test:   ");
//
//      outUnit.computeWeightChange();
//
//      if (outUnit.weightChange[0] >= -0.01220
//          && outUnit.weightChange[0] <= -0.01219
//          && outUnit.weightChange[1] >= -0.01220
//          && outUnit.weightChange[1] <= -0.01219
//          && outUnit.weightChange[2] >= -0.01220
//          && outUnit.weightChange[2] <= -0.01219) {
//        System.out.print("PASSED \n");
//      } else {
//        System.out.print("FAILED \n");
//      }
//
//      System.out.print("   computeWeightChangeMomentum Test:   ");
//
//      outUnit.computeWeightChangeMomentum();
//
//      if (outUnit.weightChangeMomentum[0] >= -0.01220*Net.momentum
//          && outUnit.weightChangeMomentum[0] <= -0.01219*Net.momentum
//          && outUnit.weightChangeMomentum[1] >= -0.01220*Net.momentum
//          && outUnit.weightChangeMomentum[1] <= -0.01219*Net.momentum
//          && outUnit.weightChangeMomentum[2] >= -0.01220*Net.momentum
//          && outUnit.weightChangeMomentum[2] <= -0.01219*Net.momentum) {
//        System.out.print("PASSED \n");
//      } else {
//        System.out.print("FAILED \n");
//      }
//
//      System.out.print("   updateWeights() Test:         ");
//
//      outUnit.updateWeights();
//
//      if (outUnit.inWeights[0] >= 0.48780 && outUnit.inWeights[0] <= 0.48781
//          && outUnit.inWeights[1] >= 0.48780 && outUnit.inWeights[1] <= 0.48781
//          && outUnit.inWeights[2] >= 0.48780 && outUnit.inWeights[2] <= 0.48781) {
//        System.out.print("PASSED \n");
//      } else {
//        System.out.print("FAILED \n");
//      }
//
//      System.out.print("\nTesting Net.java Class: \n");
//      System.out.print("   Net constructor Test:         ");
//
//      Net n;
//      n = createAnd();
//
//      boolean netCreated = false;
//      if (n.outUnit.in.contains(n.inUnit1) && n.outUnit.in.contains(n.inUnit2)
//          && n.outUnit.in.contains(n.Bias)) {
//        netCreated = true;
//      }
//      if (netCreated) {
//        System.out.print("PASSED \n");
//      } else {
//        System.out.print("FAILED \n");
//      }
//
//      System.out.print("   feedforward() Test:           ");
//
//      n.outUnit.inWeights[0] = 0.5;
//      n.outUnit.inWeights[1] = 0.5;
//      n.outUnit.inWeights[2] = 0.5;
//
//      n.feedforward(data[1]);
//
//      if (n.outUnit.activation >= 0.7310 && n.outUnit.activation <= 0.7311) {// 1
//                                                                          // /
//                                                                          // (1.0
//                                                                          // +
//                                                                          // Math.exp(1.5)
//                                                                          // ) )
//                                                                          // {
//        System.out.print("PASSED \n");
//      } else {
//        System.out.print("FAILED with activation " + n.outUnit.activation
//            + " expecting activation 0.7310 \n");
//      }
//
//      System.out.print("   computeError() Test1:         ");
//      n.outUnit.inWeights[0] = 20;
//      n.outUnit.inWeights[1] = 20;
//      n.outUnit.inWeights[2] = -30;
//
//      double err = n.computeError();
//
//      if (err > 0.0 && err < 0.0001) {
//        System.out.print("PASSED \n");
//      } else {
//        System.out.print("FAILED \n");
//      }
//
//      System.out.print("                  Test2:         ");
//
//      n.outUnit.inWeights[0] = -20;
//      n.outUnit.inWeights[1] = -20;
//      n.outUnit.inWeights[2] = 30;
//      err = n.computeError();
//
//      if (err > 1.99 && err < 2.0) {
//        System.out.print("PASSED \n");
//      } else {
//        System.out.print("FAILED \n");
//      }
//
//      System.out.print("\n   AND train() Test:             ");
//
//      n.outUnit.inWeights[0] = 0.5;
//      n.outUnit.inWeights[1] = 0.5;
//      n.outUnit.inWeights[2] = 0.5;
//      n.train();
//
//      if (n.outUnit.inWeights[0] + n.outUnit.inWeights[2] < 0
//          && n.outUnit.inWeights[1] + n.outUnit.inWeights[2] < 0
//          && n.outUnit.inWeights[0] + n.outUnit.inWeights[1]
//              + n.outUnit.inWeights[2] > 0) {
//        System.out.print("PASSED \n\n");
//      } else {
//        System.out.print(n.toString());
//        System.out.print("FAILED \n\n");
//      }
//
//      System.out.print("   OR train() Test:              ");
//
//      n = createOr();
//      n.train();
//
//      if (n.outUnit.inWeights[0] + n.outUnit.inWeights[2] > 0
//          && n.outUnit.inWeights[1] + n.outUnit.inWeights[2] > 0
//          && n.outUnit.inWeights[2] < 0) {
//        System.out.print("PASSED \n\n");
//      } else {
//        System.out.print(n.toString());
//        System.out.print("FAILED \n\n");
//      }
//    } else if (argv.length == 5) {
//
//      int[][][] trainingData;
//      try {
//        trainingData = loadTrainingDataFromFile(argv[0]);
//      } catch (Exception e) {
//        System.out.println(">>>>>>>>>>>>>>>> Failed to load: "
//            + e.getMessage());
//        return;
//      }
//
//      Net n = new Net(trainingData[0], trainingData[1]);
//      n.setTrainingParameters(Integer.parseInt(argv[1]), Double.parseDouble(argv[2]),
//          Double.parseDouble(argv[3]), Double.parseDouble(argv[4]));
//      try {
//        n.train();
//      } catch (Exception e) {
//        System.out.println(">>>>>>>>>>>>>>>>> Failed to train: " + e.getMessage());
//        return;
//      }
//
//    } else {
//      System.err.println("Invalid argument count");
//      return;
//    }
//
//    // System.out.print("<S1> Weights for " + argv[0] + ": \n");
//
//    /*
//     * // dump the activations for each input for (int r = 0; r<n.trainingData.length;
//     * r++) { n.feedforward(n.trainingData[r]); System.out.println("Activations
//     * (pat #"+(r+1)+" err "+
//     * Net.d2s(n.computeCurrentError(n.trainingTargets[r]))+")");
//     * System.out.print(n.dumpActivations()); };
//     */
//
//  }
//  
//  /**
//   * Load training data out of a file.
//   * 
//   * @param filename
//   */
//  public static int[][][] loadTrainingDataFromFile(String filename) {
//    int[][] trainingData = null;
//    int[][] trainingTargets = null;
//    try {
//      BufferedReader in = new BufferedReader(new FileReader(filename));
//      String s;
//      int state = 0, count=0;
//      int nData = 0,
//        nIn = 0,
//        nOut = 0;
//      while ((s=in.readLine())!=null) {
//        String[] parts = s.split(" ");
//        switch(state) {
//        case 0:
//          if (parts[0].equals("DATA_DESCRIPTION"))
//            state=1;
//          break;
//        case 1:
//          nData = Integer.parseInt(parts[0]);
//          nIn = Integer.parseInt(parts[1]);
//          nOut = Integer.parseInt(parts[2]);
//          trainingData = new int[nData][nIn];
//          trainingTargets = new int[nData][nOut];
//          state=2;
//          break;
//        case 2:
//          if (parts[0].equals("DATA"))
//            state=3;
//          break;
//        case 3:
//          if (parts.length!=nIn+nOut+1)
//            throw new Exception("failed reading file");
//          trainingData[count] = parseNInts(parts,0,nIn);
//          trainingTargets[count] = parseNInts(parts,nIn+1,nOut);
//          count++;
//          break;
//        }
//      }
//    } catch (Exception e) {
//      throw new Error("failed reading file "+filename);
//    }
//    int[][][] result = {trainingData,trainingTargets};
//    return result;
//  }
// 
//  
//  /**
//   *  Parse an array of strings into an array of doubles.
//   * 
//   * @param arr
//   * @return
//   */
//  public static double[] parseDoubleArray(String[] arr) {
//    double[] d = new double[arr.length];
//    for (int i=0; i<arr.length; i++) {
//      d[i] = Double.parseDouble(arr[i]);
//    }
//    return d;
//  }
//  
//  /** Parse part of an array of integers.
//   * 
//   * @param arr The array to parse
//   * @param start The element in the array to start with
//   * @param count The number of elements to parse
//   * @return the parsed integers
//   */
//  public static int[] parseNInts(String[] arr, int start, int count) {
//    int[] d = new int[count];
//    for (int i=0; i<count; i++) {
//      d[i] = Integer.parseInt(arr[i+start]);
//    }
//    return d;
//  }
//}
