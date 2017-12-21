package upmc.ri.bin;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import upmc.ri.struct.DataSet;
import upmc.ri.struct.Evaluator;
import upmc.ri.struct.STrainingSample;
import upmc.ri.struct.instantiation.MultiClass;
import upmc.ri.struct.instantiation.MultiClassHier;
import upmc.ri.struct.model.LinearStructModel_Ex;
import upmc.ri.struct.training.ITrainer;
import upmc.ri.struct.training.SGDTrainer;
import upmc.ri.utils.CSVExporter;

public class MultiClassClassif {
	
	public static void main(String[] args) {
		
		//================================================================================
	    // Setting shared data for both 0/1 and Hierarchical learning
	    //================================================================================
		// Data set
		System.out.println("Loading data");
		String path = "/home/sebastien/data_science/DAC/Master_DAC/RI/RI/2-media/data";
		List<String> files = Arrays.asList(path+"/tree-frog.txt", path+"/harp.txt",path+"/minivan.txt", 
				path+"/taxi.txt", path+"/acoustic_guitar.txt", path+"/ambulance.txt", 
				path+"/electric_guitar.txt", path+"/european_fire_salamander.txt", path+"/wood-frog.txt" ); 
		DataSet<double[],String> dataset = VisualIndexes.createDataSet(files);
		// Learning hyper parameters
		float lambda = (float) Math.pow(10,-6);
		float gama = (float) Math.pow(10,-2);
		int iterations = 10;

		//================================================================================
	    // 0/1 model
	    //================================================================================
		MultiClass instance = new MultiClass(); 
		int dim = instance.getDim();
		int classNumbers = instance.getSet().size();
		LinearStructModel_Ex<double[],String> model = new LinearStructModel_Ex<double[],String> (dim * classNumbers);
		model.setInstance(instance);
		
		Evaluator<double[],String> evaluator = new Evaluator<double[], String>();
		evaluator.setListtrain(dataset.getTrain());
		System.out.println(dataset.getTest().size());
		evaluator.setListtest(dataset.getTest());
		evaluator.setModel(model);
				
		// TODO print the right Errors in SGDTrain.train (Evaluator.evaluate)
		ITrainer<double[],String> trainer = new SGDTrainer<double[],String>(iterations, gama, lambda, evaluator);
		
		double[][] error;
		error = trainer.train(dataset.getTrain(), model);
		
		CSVExporter.exportMatrix(error, "error.txt");
		
		// Inference and evaluation (Confusion Matrix)
		List<String> trueTestLabels = new ArrayList<String>();
		List<String> predictTestLabels = new ArrayList<String>();
		for(STrainingSample<double[], String> ts : dataset.getTest()) {
			trueTestLabels.add(ts.output);
			predictTestLabels.add(model.predict(ts.input));
		}
		// TODO print all class mapping String Index to interpret confusion matrix
		System.out.println("Corresponding class and indexes");
		Iterator<Entry<String, Integer>> it = instance.getMap().entrySet().iterator();
	    while (it.hasNext()) {
	        Map.Entry pair = (Map.Entry)it.next();
	        System.out.println(pair.getKey() + " = " + pair.getValue());
	    }
        double[][] matrix ;
		matrix = instance.confusionMatrix(predictTestLabels, trueTestLabels);
		//CSVExporter.exportMatrix(matrix, "confusion.txt");
		CSVExporter.exportMatrix(matrix, "confusion_"+iterations+"_iterations"+gama+"_learningRate.txt");
		// TODO conclusion about learning how do errors spread across differents classes ?
		// TODO Display some misclassified pictures
		
		//================================================================================
	    //  Semantical Hierarchical model
	    //================================================================================
		MultiClass instanceHier = new MultiClassHier();
		int classNumbersHier = instanceHier.getSet().size();
		int dimHier = instanceHier.getDim();
		
		LinearStructModel_Ex<double[],String> modelHier = new LinearStructModel_Ex<double[],String> (dimHier * classNumbersHier);
		modelHier.setInstance(instanceHier);
		
		Evaluator<double[],String> evaluatorHier = new Evaluator<double[], String>();
		evaluatorHier.setListtrain(dataset.getTrain());
		evaluatorHier.setListtest(dataset.getTest());
		evaluatorHier.setModel(modelHier);
		
		
		// Here we evaluate on the HierDelta with params learned on HierDelta
		ITrainer<double[],String> trainerHier = new SGDTrainer<double[],String>(iterations, gama, lambda,evaluatorHier );
		double[][] errorHier;
		errorHier = trainerHier.train(dataset.getTrain(), modelHier);
		
		CSVExporter.exportMatrix(errorHier, "errorHier.txt");
		
		// Here we evaluate on the 0/1 with params learned on HierDelta
		// TODO re use parameters found on Hierarchical model to predict with 0/1 Loss model and print confusion matrix
		MultiClass instance2 = new MultiClass(); 
		modelHier.setInstance(instance2);
		evaluator.setModel(modelHier);
		evaluator.evaluate();
		// TODO see how confusion matrix work on switching params from 0/1 to Hier
		double[][] matrixHier ;
		matrixHier = instance2.confusionMatrix(predictTestLabels, trueTestLabels);
		
		//CSVExporter.exportMatrix(matrix, "confusionHier.txt");
		CSVExporter.exportMatrix(matrix, "confusionHier_"+iterations+"_iterations"+gama+"_learningRate.txt");
		
	}

}
