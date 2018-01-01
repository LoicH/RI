package upmc.ri.struct;

import java.util.List;

import upmc.ri.struct.model.IStructModel;

/** Class used to compute train and test error.
 * Contains the list of training samples, and the list of test samples.
 * Contains the model used to compute the loss
 * @param <X> The type of the input of the samples
 * @param <Y> The type of the output of the samples
 */
public class Evaluator<X,Y> {
	private List<STrainingSample<X,Y>> listTrain;
	private List<STrainingSample<X,Y>> listTest;
	private IStructModel<X,Y> model;
	
	/** The last training error computed */
	private double errTrain;
	/** The last test error computed */
	private double errTest;

	/** Compute the training and test error. */
	public void evaluate(){
		this.evaluateTestError();
		this.evaluateTrainError();
	}

	/** Compute the error on the training samples.
	 * @return The computed error on the training samples
	 */
	public double evaluateTrainError(){
		this.errTrain= this.evaluateError(this.listTrain);
		return this.errTrain;
	}

	/** Compute the error on the testing samples.
	 * @return The computed error on the testing samples
	 */
	public double evaluateTestError(){
		this.errTest= this.evaluateError(this.listTest);
		return this.errTest;
	}

	public double evaluateError(List<STrainingSample<X, Y>> tsList){
		double error = 0;
		for(STrainingSample<X,Y> ts : tsList){
			Y pred = model.predict(ts);
			error += model.instantiation().delta(ts.output,pred);
		}
		error /= tsList.size();
		return error;
	}
		
	/** Retrieve the last computed train error.
	 * Warning: Does not compute the error.
	 * @return The error computed over all the training samples
	 */
	public double getTrainError() {
		return this.errTrain;
	}

	/** Retrieve the last computed test error.
	 * Warning: Does not compute the error.
	 * @return The error computed over all the testing samples
	 */
	public double getTestError() {
		return this.errTest;
	}

	public void setListTrain(List<STrainingSample<X, Y>> listTrain) {
		this.listTrain = listTrain;
	}

	public void setListTest(List<STrainingSample<X, Y>> listtest) {
		this.listTest = listtest;
	}

	public void setModel(IStructModel<X, Y> model) {
		this.model = model;
	}

}
