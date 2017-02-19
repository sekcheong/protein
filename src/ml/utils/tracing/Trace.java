package ml.utils.tracing;


public class Trace {
	
	public static boolean enabled = true;
	
	public static void Write(String format, Object...args) {
		if (!Trace.enabled) return;
		System.out.printf(format,args);		
	}
	
	public static void WriteLine(String format, Object...args) {
		if (!Trace.enabled) return;
		Trace.Write(format, args);
		System.out.println();
	}
	
	public static void Error(String format, Object...args) {
		if (!Trace.enabled) return;
		System.err.printf(format,args);
		System.err.println();
	}
}