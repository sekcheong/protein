package ml.utils;

public class Console {

	public static void log(Object msg, Object... moreMsgs) {
		System.out.print(msg);
		for (Object s : moreMsgs) {
			System.out.print(" " + s);
		}
		System.out.println();
	}


	public static void Error(Object msg, Object... moreMsgs) {
		System.out.print(msg);
		for (Object s : moreMsgs) {
			System.out.print(" " + s);
		}
		System.err.println();
	}

}