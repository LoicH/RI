package upmc.ri.bin;

import java.util.Arrays;
import java.util.List;

import upmc.ri.struct.DataSet;
import upmc.ri.struct.Evaluator;
import upmc.ri.struct.instantiation.RankingInstanciation;
import upmc.ri.struct.model.RankingStructModel;
import upmc.ri.struct.ranking.RankingFunctions;
import upmc.ri.struct.ranking.RankingOutput;
import upmc.ri.struct.training.SGDTrainer;

public class Ranking {

	public static void main(String[] args) {
		
		// Learning hyper parameters
		float lambda = (float) Math.pow(10,-6);
		float gama = (float) Math.pow(10,1);
		int iterations = 10;
				
		System.out.println("Loading data");
		String path = "/home/sebastien/data_science/DAC/Master_DAC/RI/RI/2-media/data";
		List<String> files = Arrays.asList(path+"/tree-frog.txt", path+"/harp.txt",path+"/minivan.txt", 
				path+"/taxi.txt", path+"/acoustic_guitar.txt", path+"/ambulance.txt", 
				path+"/electric_guitar.txt", path+"/european_fire_salamander.txt", path+"/wood-frog.txt" ); 
		DataSet<double[],String> dataset = VisualIndexes.createDataSet(files);
		
		String  query = new String("taxi");
		DataSet<List<double[]>,RankingOutput> datasetRanking = RankingFunctions.convertClassif2Ranking(dataset, query);

		RankingInstanciation instanceRanking = new RankingInstanciation();
		int dimRanking = instanceRanking.getDim();
		RankingStructModel<List<double[]>,RankingOutput> modelRanking = new RankingStructModel<List<double[]>,RankingOutput>(dimRanking);
		
		Evaluator<List<double[]>,RankingOutput> evaluator = new Evaluator<List<double[]>,RankingOutput>();
		evaluator.setListtrain(datasetRanking.getTrain());
		evaluator.setListtest(datasetRanking.getTest());
		evaluator.setModel(modelRanking);
		
		SGDTrainer<List<double[]>,RankingOutput> trainer = new SGDTrainer<List<double[]>,RankingOutput>(iterations, gama, lambda, evaluator);
		trainer.train(datasetRanking.getTrain(), modelRanking);
	}

}
