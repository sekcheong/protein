package ml.utils.tracing;

public class Trace {

	public static boolean enabled = true;


	public static void Write(String format, Object... args) {
		if (!Trace.enabled) return;
		System.out.printf((String) format, args);
	}


	public static void WriteLine(String format, Object... args) {
		if (!Trace.enabled) return;
		Trace.Write(format, args);
		System.out.println();
	}


	public static void log(Object msg) {
		log(msg, new Object[0]);
	}


	public static void log(Object msg, Object... moreMsgs) {
		if (!Trace.enabled) return;
		System.out.print(msg);
		for (Object s : moreMsgs) {
			System.out.print(" " + s);
		}
		System.out.println();
	}


	public static void Error(Object msg) {
		Error(msg, new Object[0]);
	}


	public static void Error(Object msg, Object... moreMsgs) {
		if (!Trace.enabled) return;
		System.out.print(msg);
		for (Object s : moreMsgs) {
			System.out.print(" " + s);
		}
		System.err.println();
	}
}