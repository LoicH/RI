package upmc.ri.bin;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import javax.imageio.ImageIO;

import upmc.ri.struct.DataSet;
import upmc.ri.struct.Evaluator;
import upmc.ri.struct.STrainingSample;
import upmc.ri.struct.instantiation.RankingInstanciation;
import upmc.ri.struct.model.RankingStructModel;
import upmc.ri.struct.ranking.RankingFunctions;
import upmc.ri.struct.ranking.RankingOutput;
import upmc.ri.struct.training.SGDTrainer;
import upmc.ri.utils.CSVExporter;
import upmc.ri.utils.Drawing;

public class Ranking {

	public static void main(String[] args) {
		
		//================================================================================
	    // Setting data
	    //================================================================================
		int DimPCA = 250;
		System.out.println("Loading data");
		String path = "data";

		Set<String> classes = upmc.ri.io.ImageNetParser.classesImageNet();
		List<String> files = new ArrayList<String> ();
		for(String c: classes){
			files.add(path + "/" + c + ".txt");
		}
		DataSet<double[],String> dataset = VisualIndexes.createDataSet(files, DimPCA);
		System.out.println("Train labels:");
		System.out.println(dataset.countTrainLabels());
		System.out.println("Test labels:");
		System.out.println(dataset.countTestLabels());
		
		//String  query = new String("taxi");
		for(String query: classes) {
			DataSet<List<double[]>,RankingOutput> datasetRanking = RankingFunctions.convertClassif2Ranking(dataset, query);
			
			// Learning hyper parameters
			float lambda = (float) Math.pow(10,-6);
			float gama = (float) Math.pow(10,1);
			int iterations = 12;
			
			RankingInstanciation instanceRanking = new RankingInstanciation();
			int dimRanking = instanceRanking.getDim();
			
			RankingStructModel modelRanking = new RankingStructModel(dimRanking);
			modelRanking.setInstance(instanceRanking);
			
			Evaluator<List<double[]>,RankingOutput> evaluator = new Evaluator<List<double[]>,RankingOutput>();
			evaluator.setListTrain(datasetRanking.getTrain());
			evaluator.setListTest(datasetRanking.getTest());
			evaluator.setModel(modelRanking);
			
			// Train model and get error evaluation
			SGDTrainer<List<double[]>,RankingOutput> trainer = new SGDTrainer<List<double[]>,RankingOutput>(iterations, gama, lambda, evaluator);
			double[][] error;
			error = trainer.train(datasetRanking.getTrain(), modelRanking);
			CSVExporter.exportMatrix(error, "error_"+query+".txt");
			
			
			RankingOutput y_train_pred = modelRanking.predict((STrainingSample<List<double[]>, RankingOutput>) datasetRanking.listTrain.get(0));
			System.out.println("Selected sample : "+ datasetRanking.listTrain.get(0));
			double AP_train = RankingFunctions.averagePrecision(y_train_pred);
			System.out.println("Average Precision (TRAIN) for "+query+" : "+AP_train);
			double[][] pr_train = RankingFunctions.recalPrecisionCurve(y_train_pred);
			BufferedImage trainPrecisionRecal = Drawing.traceRecallPrecisionCurve(y_train_pred.getNbPlus(), pr_train);
	
			RankingOutput y_test_pred = modelRanking.predict((STrainingSample<List<double[]>, RankingOutput>) datasetRanking.listTest.get(0));
			System.out.println("Selected sample : "+ datasetRanking.listTrain.get(0));
			double AP_test = RankingFunctions.averagePrecision(y_test_pred);
			System.out.println("Average Precision (TEST) for "+query+" : "+AP_test);
			double[][] pr_test = RankingFunctions.recalPrecisionCurve(y_test_pred);
			BufferedImage testPrecisionRecal = Drawing.traceRecallPrecisionCurve(y_train_pred.getNbPlus(), pr_test);
			
			File train_rp = new File("TrainPrecisionRecall" + query+".png");
			File test_rp = new File("TestPrecisionRecall-" + query+".png");
			try {
				ImageIO.write(trainPrecisionRecal, "png",  train_rp);
				ImageIO.write(testPrecisionRecal, "png", test_rp);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
	}
	}

}
