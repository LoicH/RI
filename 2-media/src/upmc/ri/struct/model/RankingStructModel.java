package upmc.ri.struct.model;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;

import upmc.ri.struct.STrainingSample;
import upmc.ri.struct.ranking.RankingFunctions;
import upmc.ri.struct.ranking.RankingOutput;
import upmc.ri.utils.VectorOperations;

public class RankingStructModel<X,Y> extends LinearStructModel<List<double[]>,RankingOutput> {

	public RankingStructModel(int dimpsi) {
		super(dimpsi);
	}

	@Override
	public RankingOutput predict(List<double[]> x) {	
		// Create map of 'input -> score'
		HashMap<double[], Float> map = new HashMap<double[], Float>();
		List<Integer> labelsGT = new ArrayList<Integer>();
		int nbPlus = 0;
		// Create list of indexes:
		List<Integer> listIdx = new ArrayList<Integer>();

		double[] param = super.getParameters();
		
		for (int i = 0; i<x.size(); i++) {
			double[] xi = x.get(i);
			double score = VectorOperations.dot(param, xi);
			map.put(xi, (float) score);
			listIdx.add(i);
			if (score > 0) {
				nbPlus ++;
				labelsGT.add(1);
			} 
			else {
				labelsGT.add(-1);
			}
		}
		// Sort 'listX' with keys from 'map'
		Collections.sort(listIdx, new Comparator<Integer>() {
			@Override
			public int compare(Integer i1, Integer i2) {
				if (map.get(x.get(i1)) > map.get(x.get(i2))){
					return -1;
				}
				else {
					return 1;
				}			}
		});
		RankingOutput result = new RankingOutput(nbPlus, listIdx, labelsGT);
		return result;
	}

	@Override
	public RankingOutput lai(STrainingSample<List<double[]>, RankingOutput> ts) {
		return RankingFunctions.loss_augmented_inference(ts, this.getParameters());
	}
	

}
