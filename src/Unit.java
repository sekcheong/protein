import java.util.*;

/**
 * Unit is the basic unit of a neural network.
 */
public class Unit {


  private List<Unit> _input;       // input Units for this Unit
  
  private List<Unit> _output;      // output Units for this Unit

  private double[] _inputWeights;  // current (input) weights for this Unit 

  private double _activation;      // this unit's activation level

  private double _error;           // this unit's error

  private double _delta;           // this unit's delta

  private double[] _weightChange;  // weight changes for each weight
  
  private double[] weightChangeMomentum;

  private int _index;              // index number for this Unit

  private Net _net;                        // network this unit belongs to

  /**
   * Constructor for Unit class. Takes an index number and Net to which Unit
   * belongs.
   * 
   */
  public Unit() {
    _input = new ArrayList<Unit>();    
    _output = new ArrayList<Unit>();
  }

  public void addIncomingUnit(Unit unit) {
    _input.add(unit);
  }

  public void setOutgoingUnit(Unit outUnit) {
    _output.add(outUnit);
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
	  return 0;
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
