package upmc.ri.index;

import java.util.List;

import upmc.ri.utils.VectorOperations;



public class VIndexFactory {
	/** Compute Bag of Words from image features
	 * @param ib The ImageFeatures object that holds the Bag of Features
	 * @return An array of frequencies for each word from 0 to 999
	 */
	public static double[] computeBow(ImageFeatures ib) {
		List<Integer> words = ib.getwords();
		double[] target = new double[ImageFeatures.tdico];
		for (int w : words) {
		    target[w] += 1;
		 }
		// Scaling:
		double norm = VectorOperations.norm2(target);
		for (int i = 0; i<target.length; i++){
			target[i] /= norm;
		}
		return target;
		
	}
}