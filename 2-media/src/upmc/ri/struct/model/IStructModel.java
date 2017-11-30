package upmc.ri.struct.model;

import upmc.ri.struct.STrainingSample;
import upmc.ri.struct.instantiation.IStructInstantiation;

public interface IStructModel<X,Y> {
	public Y predict(X x);
	public Y lai(STrainingSample<X,Y> ts);
	public IStructInstantiation <X,Y> instantiation();
	public double[] getParameters();
	public void setParameters(double[] w);
}
