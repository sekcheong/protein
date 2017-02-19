package ml.utils.console;

public class Console {

	public static void log(String msg, String... moreMsgs) {
		System.out.print(msg);
		for (String s:moreMsgs) {
			System.out.print(" " + s);
		}
	}

	public static void Error(String msg, String... moreMsgs) {
		System.out.print(msg);
		for (String s:moreMsgs) {
			System.out.print(" " + s);
		}
	}
	
	
}