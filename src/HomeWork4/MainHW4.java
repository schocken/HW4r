package HomeWork4;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import HomeWork4.Knn.EditMode;
import weka.core.Instances;

public class MainHW4 {

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	public static Instances loadData(String fileName) throws IOException {
		BufferedReader datafile = readDataFile(fileName);
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	public static void main(String[] args) throws Exception {
        //TODO: complete the Main method
		Instances cancerData = loadData("cancer.txt");
		Instances glassData = loadData("glass.txt");
		Knn classifier = new Knn();
		classifier.setEditMode(EditMode.None);
		System.out.println("Im here");
		classifier.buildClassifier(cancerData);
		System.out.println("K:" + classifier.k + " LP:" + classifier.lp + " Majority function:" + classifier.maj);
		System.out.println("Presicion:" + classifier.calcConfusion(cancerData)[0] + "Recall:" + classifier.calcConfusion(cancerData)[1]);
	}

}
