package upmc.ri.struct.training;

import java.util.ArrayList;
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

	public SGDTrainer(int iterations, double gradStep, double lambda,Evaluator<X, Y> eval ) {
		this.iterations = iterations;
		this.gradStep = gradStep;
		this.lambda = lambda;
		this.eval = eval;
	}

	
	public double[][] train(List<STrainingSample<X, Y>> lts, IStructModel<X, Y> model) {
		int N = lts.size(); // Number of samples
		//this.eval = new Evaluator<X, Y>();
		//this.eval.setListtrain(lts);
		//this.eval.setModel(model);
		
		//TODO check if w is a pointer
		double[] w = model.getParameters();
		Arrays.fill(w, 0);
		model.setParameters(w);
		
		ArrayList<Float> error = new ArrayList<Float>();
		double[][] errorTrainTest = new double[2][this.iterations];
		
		Random generator = new Random();
		for(int t = 0; t<this.iterations; t++){
			System.out.println("Iteration "+ t);
			for(int i = 0; i<N; i++){
				
//				if (i%1000 == 0) {
//					System.out.println(i + "/" + N);
//					System.out.println("Convex loss:" + convex_loss(lts, model));
//				}
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
			}
			
//			this.eval.evaluateTrain();
//			System.out.println("Train error:"+this.eval.getErr_train());
//			error.add((float) this.eval.getErr_train());
			
			this.eval.evaluate();
			System.out.println("Train error:"+this.eval.getErr_train());
			System.out.println("Test error:"+this.eval.getErr_test());
			errorTrainTest[0][t] = this.eval.getErr_train();
			errorTrainTest[1][t] = this.eval.getErr_test();
		}
		
		return errorTrainTest;
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
	
	public void setEval(Evaluator<X, Y> eval) {
		this.eval = eval;
	}

}
