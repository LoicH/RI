package upmc.ri.struct.training;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import upmc.ri.struct.Evaluator;
import upmc.ri.struct.STrainingSample;
import upmc.ri.struct.model.IStructModel;
import upmc.ri.utils.VectorOperations;

public class SGDTrainer<X, Y> implements ITrainer<X, Y> {
	
	private Evaluator<X, Y> eval ;
	
	private int iterations;
	private double gradStep;
	private double lambda;
	
	public SGDTrainer(int iterations, double gradStep, double lambda) {
		this.iterations = iterations;
		this.gradStep = gradStep;
		this.lambda = lambda;
	}

	
	public void train(List<STrainingSample<X, Y>> lts, IStructModel<X, Y> model) {
		int N = lts.size(); // Number of samples
		this.eval = new Evaluator<X, Y>();
		this.eval.setListtrain(lts);
		this.eval.setModel(model);
		
		// Init w
		//TODO check if w is a pointer
		double[] w = model.getParameters();
		Arrays.fill(w, 0);
		model.setParameters(w);
		
		Random generator = new Random();
		for(int t = 0; t<this.iterations; t++){
			for(int i = 0; i<N; i++){
				// Choose random sample
				STrainingSample<X, Y> ts = lts.get(generator.nextInt(N));
				// Compute y_hat
				Y yHat = model.lai(ts);
				// Compute gradient
				double[] grad = model.instantiation().psi(ts.input, yHat);
				double[] psi2 = model.instantiation().psi(ts.input, ts.output);
				for(int j = 0; j<grad.length; j++){
					grad[j] -= psi2[j];
				}
				// Update w and eval
				for(int j = 0; j<w.length; j++){
					w[j] -= this.gradStep * (this.lambda * w[j] + grad[j]);
				}
				model.setParameters(w);
				this.eval.evaluateTrain();
				System.out.println("Train error:"+this.eval.getErr_train());
			}
		}
		
		
	}
	
	public double convex_loss(List<STrainingSample<X, Y>> lts, IStructModel<X, Y> model){
		double P = this.lambda * 0.5 * Math.pow(VectorOperations.norm2(model.getParameters()), 2);
		double N = lts.size();
		double[] w = model.getParameters();
		
		for (int i = 0; i<N; i++){
			STrainingSample<X, Y> ts= lts.get(i);
			Y yHat = model.lai(ts);
			double tmp = model.instantiation().delta(ts.output, yHat);
			double[] psi = model.instantiation().psi(ts.input, yHat);
			tmp += VectorOperations.dot(psi, w);
			psi = model.instantiation().psi(ts.input, ts.output);
			tmp -= VectorOperations.dot(psi, w);
			P += 1.0/N * tmp;
		}
		
		return P;
	}

}
