package upmc.ri.struct.model;

import upmc.ri.struct.STrainingSample;
import upmc.ri.struct.instantiation.IStructInstantiation;

/** Representation of a machine learning model.
 * @param <X> The type of the inputs
 * @param <Y> Type of the outputs
 */
public interface IStructModel<X,Y> {
	/** Compute the output associated to an output
	 * @param x The input
	 * @return The computed output
	 */
	public Y predict(X x);
	
	public Y lai(STrainingSample<X,Y> ts);
	
	public IStructInstantiation <X,Y> instantiation();
	
	public double[] getParameters();
	
	public void setParameters(double[] w);
}
