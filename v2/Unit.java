/***************************************************
 Starter code for CS182
 Assignment 4: Backpropagation
 Unit.java
 **************************************************/

import java.util.*;

/**
 * Unit is the basic unit of a neural network.
 */
public class Unit {

  // Unit attributes

  // The following Unit attributes are suggested starting points
  // for the assignment. You may wish to modify them as necessary
  // for your particular implementation.

  Vector<Unit> in; // input Units for this Unit

  Vector<Unit> out; // output Units for this Unit

  double[] inWeights; // current (input) weights for this Unit

  // I.E. inWeights[1] is the weight from the in[1] unit.

  double activation; // this unit's activation level

  double error; // this unit's error

  double delta; // this unit's delta

  double[] weightChange; // weight changes for each weight
  
  double[] weightChangeMomentum;

  int index; // index number for this Unit

  Net net; // network this unit belongs to

  /**
   * Constructor for Unit class. Takes an index number and Net to which Unit
   * belongs.
   * 
   */
  public Unit() {
    in = new Vector<Unit>();
    out = new Vector<Unit>();
  }

  public void addIncomingUnit(Unit inUnit) {
    this.in.add(inUnit);
  }

  public void setOutgoingUnit(Unit outUnit) {
    this.out.add(outUnit);
  }

  /**
   * initalize() Randomize all incoming weights between the network's minimum
   * and maximum weights, including bias weights.
   */
  public void initialize() {
    activation = 0;
    inWeights = new double[in.size()];
    weightChange = new double[in.size()];
    weightChangeMomentum = new double[in.size()];
    for (int i=0; i<inWeights.length; i++)
      inWeights[i] = getRandom(Net.MIN_WEIGHT,Net.MAX_WEIGHT);
  }

  /**
   * computeActivation() Compute output activation of this unit. Apply sigmoid
   * function to weighted sum of inputs.
   */
  public double computeActivation() {
    // compute weighted sum of inputs
    double sum=0;
    for (int i=0; i<in.size(); i++) {
      sum += in.get(i).activation * inWeights[i];
    }
    // apply sigmoid
    activation = 1.0 / (1+Math.exp(-sum));
    return activation;
  }

  /**
   * computeError(targets)
   * 
   * Computes error and delta for the output node. (not squared)
   * 
   * @param target
   *          is the current true output
   */
  public void computeError(int target) {
    error = target - activation;
    delta = activation * (1-activation) * error;
  }

  /**
   * computeWeightChange()
   * 
   * Calculate the current weight change, including a momentum factor. May want
   * to use computeWeightChangeMomentum to calculate the momentum factor.
   */
  public void computeWeightChange() {
    computeWeightChangeMomentum();
    for (int i=0; i<inWeights.length; i++) {
      double wc = -Net.learningRate * -in.get(i).activation * delta;
      weightChange[i] = wc + weightChangeMomentum[i];
    }
  }

  /**
   * Update changes to weights for this pattern.
   */
  public void updateWeights() {
    for (int i=0; i<inWeights.length; i++) {
      inWeights[i] += weightChange[i];
    }
  }

  /**
   * Calculate momentum factor for weight change. Store in this.weightChangeMomentum.
   */
  public void computeWeightChangeMomentum() {
    for (int i=0; i<inWeights.length; i++) {
      weightChangeMomentum[i] = weightChange[i] * Net.momentum;
    }
  }

  // additional methods?
  // ** to be filled in **

  /**
   * Return a random number between min and max.
   */
  Random randy = new Random();

  public double getRandom(double min, double max) {
    return (randy.nextDouble() * (max - min)) + min;
  }
}
