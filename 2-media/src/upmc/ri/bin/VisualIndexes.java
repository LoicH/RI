package upmc.ri.bin;

import java.util.ArrayList;
import java.util.List;

import upmc.ri.index.ImageFeatures;
import upmc.ri.index.VIndexFactory;
import upmc.ri.io.ImageNetParser;
import upmc.ri.struct.DataSet;
import upmc.ri.struct.STrainingSample;

public class VisualIndexes {
	public static DataSet<double[], String> createDataSet(List<String> filenames){
		List<STrainingSample<double[], String>> trainSamples = new ArrayList<STrainingSample<double[], String>>();
		List<STrainingSample<double[], String>> testSamples  = new ArrayList<STrainingSample<double[], String>>();
		// TODO Remove magic number: 800 = number of training samples
		int threshold = 800;
		for (String filename : filenames){
			System.out.println("Retrieving features from "+filename);
			int i = 0;
			try {
				List<ImageFeatures> featuresList = ImageNetParser.getFeatures(filename);
				for (ImageFeatures feat : featuresList){
					double[] bow = VIndexFactory.computeBow(feat);
					//TODO Remove .txt from filename
					STrainingSample<double[], String> sample = new STrainingSample<double[], String>(bow, filename);
					if (i<threshold) {
						trainSamples.add(sample);
					}
					else {
						testSamples.add(sample);
					}
					i += 1;
				}
			} catch (Exception e) {
				System.out.println("Failed to retrieve features for "+filename);
				e.printStackTrace();
			}
		}
		return new DataSet<double[], String>(trainSamples, testSamples);
	}

}
