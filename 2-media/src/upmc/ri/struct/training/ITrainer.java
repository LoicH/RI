package upmc.ri.struct.training;

import java.util.List;
import upmc.ri.struct.STrainingSample;
import upmc.ri.struct.model.IStructModel;

public interface ITrainer<X,Y> {
	/** Train the model
	 * @param lts List of training samples
	 * @param model The model
	 * @return double[2][iterations number]: first line is the train error, 
	 * 		and the second is the test error 
	 */
	public double[][]  train(List<STrainingSample<X, Y>> lts , IStructModel<X,Y> model);

}
