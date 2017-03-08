package ml.utils.tracing;

public class Trace {

	public static void printf(String format, Object... args) {
		System.out.printf((String) format, args);
	}


	public static void log(Object msg) {
		log(msg, new Object[0]);
	}


	public static void log(Object msg, Object... moreMsgs) {

		System.out.print(msg);
		for (Object s : moreMsgs) {
			System.out.print(s);
		}
		System.out.println();
	}


	public static void traceError(Object msg) {
		TraceError(msg, new Object[0]);
	}


	public static void TraceError(Object msg, Object... moreMsgs) {
		System.err.print(msg);
		for (Object s : moreMsgs) {
			System.err.print(s);
		}
		System.err.println();
	}
}