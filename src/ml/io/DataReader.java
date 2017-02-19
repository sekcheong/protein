package ml.io;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class DataReader {

	private BufferedReader _buffReader = null;

	public DataReader(String fileName) throws FileNotFoundException {
		FileReader freader = new FileReader(fileName);
		_buffReader = new BufferedReader(freader);
	}

	public void Read() throws IOException {
		List<List<String>> proteins = new ArrayList<List<String>>();
		List<String> aminos = null;

		int lineNo = 0;
		int count = 0;

		while (true) {
			String line;

			line = _buffReader.readLine();
			lineNo++;
			
			if (line == null) {
				if (aminos != null) {
					proteins.add(aminos);
				}
				break;
			}
			
			line = line.trim();
			
			if (line.startsWith("#")) {
				continue;
			}

			if (line.startsWith("<>")) {
				count++;
				if (aminos != null) {
					proteins.add(aminos);
				}
				aminos = new ArrayList<String>();
				continue;
			}
			
			if (line.toUpperCase().startsWith("END") || line.toUpperCase().startsWith("<END>")) {
				continue;
			}

			if (line.length() > 0) {
				aminos.add(line);
			}
			else {
				System.out.println("Line: " + lineNo + " is blank.");
			}

		}

		
//		int total = 0;
//		for (int i = 0; i < examples.size(); i++) {
//			System.out.println(i + ": " + examples.get(i).size() + examples.get(i));
//			total = total + examples.size();
//		}
//
//		System.out.println("#" + examples.size() + ": " + count + " : " + total);

	}
}
