package upmc.ri.struct.model;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.TreeMap;

import upmc.ri.struct.STrainingSample;
import upmc.ri.struct.ranking.RankingOutput;
import upmc.ri.utils.VectorOperations;

public class RankingStructModel<X,Y> extends LinearStructModel<List<double[]>,RankingOutput> {

	public RankingStructModel(int dimpsi) {
		super(dimpsi);
	}

	@Override
	public RankingOutput predict(List<double[]> x) {
		// TODO 
//		List orderedList = new ArrayList(x.size());
//		for (double[] x1 : x){
//			double val = VectorOperations.dot(super.getParameters(), x1);
//			orderedList.add(val);
//		}
//		Collections.sort(orderedList);
//		RankingOutput ranking = new RankingOutput(0, orderedList, orderedList);
//		getPositionningFromRanking
		
		// Create map of 'input -> score'
		TreeMap<double[], Float> map = new TreeMap<double[], Float>();
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
		// TODO Auto-generated method stub
		return null;
	}
	

}
