package upmc.ri.main;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import upmc.ri.bin.VisualIndexes;
import upmc.ri.index.ImageFeatures;
import upmc.ri.index.VIndexFactory;
import upmc.ri.io.ImageNetParser;
import upmc.ri.struct.DataSet;
import upmc.ri.utils.PCA;

public class Main {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		System.out.println("Start");
		List<String> files = Arrays.asList("data/tree-frog.txt", "data/harp.txt");
		DataSet<double[], String> ds = VisualIndexes.createDataSet(files);
		System.out.println("DataSet created");
		
		System.out.println("Retrieving n01644373_354");
		List<ImageFeatures> features = null;
		try {
			 features = ImageNetParser.getFeatures("data/tree-frog.txt");
		} catch (Exception e) {
			System.out.println("Can't read file tree-frog.txt");
			e.printStackTrace();
			System.exit(1);
		}

		ImageFeatures target = null;
		for(ImageFeatures feat : features) {
			if (feat.getiD().equals("n01644373_354")) {
				target = feat;
				System.out.println("Found target");
				break;
			}
		}
		double[] words = VIndexFactory.computeBow(target);
		System.out.println(words);
		//TODO Plot histogram
		for(int i = 0; i < words.length; i++){
			System.out.println(i+":"+words[i]);
		}
		
		System.out.println("Computing PCA");
		DataSet<double[], String> pca = PCA.computePCA(ds, 250);
		
		System.out.println("Over");
	}

}
