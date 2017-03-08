package ml.utils;

public class Console {

	public static void writeLine(String msg) {
		writeLine(msg, new Object[0]);
	}


	public static void writeLine(Object msg, Object... moreMsgs) {
		write(msg, moreMsgs);
		System.out.println();
	}


	public static void write(Object msg, Object... moreMsgs) {
		System.out.print(msg);
		for (Object s : moreMsgs) {
			System.out.print(" " + s);
		}
	}

}