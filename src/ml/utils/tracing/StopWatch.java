package ml.utils.tracing;

public class StopWatch {
	private long _startTime;
	private long _elapsedTime;


	public StopWatch() {
		_startTime = System.currentTimeMillis();
	}


	public static StopWatch start() {
		return new StopWatch();
	}


	public void stop() {
		_elapsedTime = System.currentTimeMillis() - _startTime;
	}


	public double elapsedTime() {
		return ((double) _elapsedTime) / 1000;
	}
}
