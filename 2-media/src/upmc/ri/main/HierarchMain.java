package upmc.ri.main;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import upmc.ri.bin.VisualIndexes;
import upmc.ri.struct.DataSet;
import upmc.ri.struct.Evaluator;
import upmc.ri.struct.STrainingSample;
import upmc.ri.struct.instantiation.MultiClass;
import upmc.ri.struct.instantiation.MultiClassHier;
import upmc.ri.struct.model.LinearStructModel_Ex;
import upmc.ri.struct.training.ITrainer;
import upmc.ri.struct.training.SGDTrainer;
import upmc.ri.utils.CSVExporter;

/** Main used to test the hierarchical classification.
 */
public class HierarchMain {

	public static void main(String[] args) {
		//================================================================================
	    // Setting data
	    //================================================================================
		System.out.println("Loading data");
		String path = "data";

		Set<String> classes = upmc.ri.io.ImageNetParser.classesImageNet();
		List<String> files = new ArrayList<String> ();
		for(String c: classes){
			files.add(path + "/" + c + ".txt");
		}
		DataSet<double[],String> dataset = VisualIndexes.createDataSet(files);
		System.out.println("Train labels:");
		System.out.println(dataset.countTrainLabels());
		System.out.println("Test labels:");
		System.out.println(dataset.countTestLabels());
		// Learning hyper parameters
		double lambda = Math.pow(10,-4);
		double gamma = Math.pow(10,-2);
		int iterations = 25;
		
		//================================================================================
	    //  Semantical Hierarchical model
	    //================================================================================
		MultiClass instanceHier = new MultiClassHier();
		int classNumbersHier = instanceHier.getSet().size();
		int dimHier = instanceHier.getDim();
		
		LinearStructModel_Ex<double[],String> model = new LinearStructModel_Ex<double[],String> (dimHier * classNumbersHier);
		model.setInstance(instanceHier);
		
		Evaluator<double[],String> evaluator = new Evaluator<double[], String>();
		evaluator.setListTrain(dataset.getTrain());
		evaluator.setListTest(dataset.getTest());
		evaluator.setModel(model);
		
		
		ITrainer<double[],String> trainerHier = new SGDTrainer<double[],String>(iterations, gamma, lambda, evaluator);
		double[][] errorHier;
		errorHier = trainerHier.train(dataset.getTrain(), model);
		
		CSVExporter.exportMatrix(errorHier, "errorHier.txt");
				
		List<String> trueTestLabels = new ArrayList<String>();
		List<String> predictTestLabels = new ArrayList<String>();
		for(STrainingSample<double[], String> ts : dataset.getTest()) {
			trueTestLabels.add(ts.output);
			String pred = model.predict(ts.input);
			predictTestLabels.add(pred);
		}

		double[][] matrix = instanceHier.confusionMatrix(predictTestLabels, trueTestLabels);
		//CSVExporter.exportMatrix(matrix, "confusionHier.txt");
		CSVExporter.exportMatrix(matrix, "confusionHier_"+iterations+"_iterations"+gamma+"_learningRate.txt");
		instanceHier.showConfMatrix(matrix);
			
	}

}
