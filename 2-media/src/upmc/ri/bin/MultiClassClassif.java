package upmc.ri.bin;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import upmc.ri.struct.DataSet;
import upmc.ri.struct.Evaluator;
import upmc.ri.struct.STrainingSample;
import upmc.ri.struct.instantiation.MultiClass;
import upmc.ri.struct.model.LinearStructModel_Ex;
import upmc.ri.struct.training.ITrainer;
import upmc.ri.struct.training.SGDTrainer;

public class MultiClassClassif {

	public static void main(String[] args) {
		// Chargement des données
		System.out.println("Loading data");
		String path = "/home/sebastien/data_science/DAC/Master_DAC/RI/RI for image/data";
		List<String> files = Arrays.asList(path+"/tree-frog.txt", path+"/harp.txt",path+"/minivan.txt", path+"/taxi.txt" );
		DataSet<double[],String> dataset = VisualIndexes.createDataSet(files);

		MultiClass instance = new MultiClass(); 
		int dim = 250;
		//int classNumbers = instance.set.size();
		int classNumbers = 9;
		LinearStructModel_Ex<double[],String> model = new LinearStructModel_Ex<double[],String> (dim * classNumbers);
		model.setInstance(instance);
		
		// Création d'un évaluateur
		Evaluator<double[],String> evaluator = new Evaluator<double[], String>();
		evaluator.setListtrain(dataset.getTrain());
		evaluator.setListtest(dataset.getTest());
		evaluator.setModel(model);
		
		// Apprentissage du modèle
		float lambda = (float) Math.pow(10,-6);
		float gama = (float) Math.pow(10,-2);
		int iterations = 100;
		// TODO modifier SGDTrainer pour printer ConvexLoss de temps en temps
		ITrainer<double[],String> trainer = new SGDTrainer<double[],String>(iterations, gama, lambda);
		trainer.train(dataset.getTrain(), model);
		
		List<String> trueTestLabels = new ArrayList<String>();
		List<String> predictTestLabels = new ArrayList<String>();
		for(STrainingSample<double[], String> ts : dataset.getTest()) {
			trueTestLabels.add(ts.output);
			predictTestLabels.add(model.predict(ts.input));
		}
				
		
		instance.confusionMatrix(predictTestLabels, trueTestLabels);
		
	}

}
