package upmc.ri.struct.model;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.stream.IntStream;

import upmc.ri.struct.STrainingSample;
import upmc.ri.struct.ranking.RankingFunctions;
import upmc.ri.struct.ranking.RankingOutput;
import upmc.ri.utils.VectorOperations;

public class RankingStructModel extends LinearStructModel<List<double[]>,RankingOutput> {

	public RankingStructModel(int dimpsi) {
		super(dimpsi);
	}
	@Override
	public RankingOutput predict(STrainingSample<List<double[]>, RankingOutput> ts) {	
		
		List<Double> scores = new ArrayList<Double>(ts.input.size());
		double[] param = super.getParameters();
		// Get score from each image 
		for (double[] image : ts.input) {
			scores.add(VectorOperations.dot(param, image));
		}
		// Sort the list of scores to get the ranking
		int[] sorted_index = IntStream.range(0,scores.size()).boxed().sorted((i,j) -> (scores.get(i) <= scores.get(j)?1:-1)).mapToInt(ele -> ele).toArray();
		List<Integer> ranking = new ArrayList<Integer>(sorted_index.length);
		for (int rank : sorted_index) {
			ranking.add(rank);
		}
		return new RankingOutput(ts.output.getNbPlus(),ranking, ts.output.getLabelsGT());
		
// DOES  NOT WORK, ALWAYS GIVE 0 on train and test error
//		// Create map of 'input -> score'
//		final HashMap<double[], Float> map = new HashMap<double[], Float>();
//		List<Integer> labelsGT = new ArrayList<Integer>();
//		int nbPlus = 0;
//		// Create list of indexes:
//		List<Integer> listIdx = new ArrayList<Integer>();
//		double[] param = super.getParameters();
//		for (int i = 0; i<ts.input.size(); i++) {
//			double[] xi = ts.input.get(i);
//			double score = VectorOperations.dot(param, xi);
//			map.put(xi, (float) score);
//			listIdx.add(i);
//			if (score > 0) {
//				nbPlus ++;
//				labelsGT.add(1);
//			} 
//			else {
//				labelsGT.add(-1);
//			}
//		}
//		// Sort 'listX' with keys from 'map'
//		final List<double[]> finalX = new ArrayList<double[]>(ts.input);
//		
//		Collections.sort(listIdx, new Comparator<Integer>() {
//			public int compare(Integer i1, Integer i2) {
//				if (map.get(finalX.get(i1)) > map.get(finalX.get(i2))){
//					return -1;
//				}
//				else {
//					return 1;
//				}			}
//		});
//		RankingOutput result = new RankingOutput(nbPlus, listIdx, labelsGT);
//		return result;
	}

	public RankingOutput lai(STrainingSample<List<double[]>, RankingOutput> ts) {
		return RankingFunctions.loss_augmented_inference(ts, this.getParameters());
	}
	

}
