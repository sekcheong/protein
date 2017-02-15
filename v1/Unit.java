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
    // ** to be filled in **
  }

  /**
   * computeActivation() Compute output activation of this unit. Apply sigmoid
   * function to weighted sum of inputs.
   */
  public double computeActivation() {
    // ** to be filled in **
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
    // ** to be filled in **
  }

  /**
   * computeWeightChange()
   * 
   * Calculate the current weight change, including a momentum factor. May want
   * to use computeWeightChangeMomentum to calculate the momentum factor.
   */
  public void computeWeightChange() {
    // ** to be filled in **
  }

  /**
   * Update changes to weights for this pattern.
   */
  public void updateWeights() {
    // ** to be filled in **
  }

  /**
   * Calculate momentum factor for weight change. Store in this.weightChangeMomentum.
   */
  public void computeWeightChangeMomentum() {
    // ** to be filled in **
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
