package upmc.ri.main;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;

import upmc.ri.bin.VisualIndexes;
import upmc.ri.struct.DataSet;
import upmc.ri.struct.Evaluator;
import upmc.ri.struct.STrainingSample;
import upmc.ri.struct.instantiation.MultiClass;
import upmc.ri.struct.model.LinearStructModel_Ex;
import upmc.ri.struct.training.ITrainer;
import upmc.ri.struct.training.SGDTrainer;
import upmc.ri.utils.CSVExporter;
import upmc.ri.utils.MatrixOperations;
import upmc.ri.utils.PCA;

/** Main used to test the hierarchical classification.
 */
public class MultiClassMain {

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


	int iterations = 10;
	double lambda = Math.pow(10,-2);
 	double gamma = Math.pow(10,-6);

	//================================================================================
    // 0/1 model
    //================================================================================
	MultiClass instance = new MultiClass(DimPCA); 
	int classNumbers = instance.getSet().size();
	LinearStructModel_Ex<double[],String> model = new LinearStructModel_Ex<double[],String> (DimPCA * classNumbers);
	model.setInstance(instance);
	
	Evaluator<double[],String> evaluator = new Evaluator<double[], String>();
	evaluator.setListTrain(dataset.getTrain());
	System.out.println(dataset.getTest().size());
	evaluator.setListTest(dataset.getTest());
	evaluator.setModel(model);
			
	// TODO print the right Errors in SGDTrain.train (Evaluator.evaluate)
	ITrainer<double[],String> trainer = new SGDTrainer<double[],String>(iterations, gamma, lambda, evaluator);
	
	double[][] error;
	error = trainer.train(dataset.getTrain(), model);
	
	CSVExporter.exportMatrix(error, "error.txt");
	
	// Inference and evaluation (Confusion Matrix)
	List<String> trueTestLabels = new ArrayList<String>();
	List<String> predictTestLabels = new ArrayList<String>();
	for(STrainingSample<double[], String> ts : dataset.getTest()) {
		trueTestLabels.add(ts.output);
		predictTestLabels.add(model.predict(ts));
	}
	// TODO print all class mapping String Index to interpret confusion matrix
	System.out.println("Corresponding class and indexes");
	Iterator<Entry<String, Integer>> it = instance.getMap().entrySet().iterator();
    while (it.hasNext()) {
        Entry<String, Integer> pair = it.next();
        System.out.println(pair.getKey() + " = " + pair.getValue());
    }
    double[][] matrix ;
	matrix = instance.confusionMatrix(predictTestLabels, trueTestLabels);
	double[][]normMat = MatrixOperations.normalizeCol(matrix);
	//CSVExporter.exportMatrix(matrix, "confusion.txt");
	CSVExporter.exportMatrix(normMat, "confusionNorm_"+iterations+"_iterations"+gamma+"_learningRate.txt");
	// TODO conclusion about learning how do errors spread across differents classes ?
	// TODO Display some misclassified pictures
	instance.showConfMatrix(matrix);
	
	// Grid search:
//	iterations = 10;
//	List<Double> lambdaVal = new ArrayList<Double>();
//	List<Double> gammaVal = new ArrayList<Double>();
//	
//	for(int i = -10; i<=0; i++){
//		lambdaVal.add(Math.pow(10, i));
//		gammaVal.add(Math.pow(10, i));
//	}
//	
//	double bestLambda = -1;
//	double bestGamma = -1;
//	double bestError = Double.POSITIVE_INFINITY;
//	for(double l:lambdaVal){
//		for(double g:gammaVal){
//			System.out.print("Testing lambda="+l+", gamma="+g+"... ");
//			trainer = new SGDTrainer<double[],String>(iterations, g, l, evaluator);
//			double [] trainError = trainer.train(dataset.getTrain(), model)[0];
//			double gridTrainError = trainError[iterations-1];
//			System.out.println("Score="+gridTrainError);
//			if (Double.compare(gridTrainError, bestError) < 0){
//				bestLambda = l;
//				bestGamma = g;
//				bestError = gridTrainError;
//			}
//		}
//	}
//	System.out.println("Best score: "+ bestError);
//	System.out.println("Best lambda: "+ bestLambda);
//	System.out.println("Best gamma: "+ bestGamma);
		
	}

}
