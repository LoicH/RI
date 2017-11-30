package upmc.ri.struct.model;

import java.util.Arrays;

import upmc.ri.struct.STrainingSample;
import upmc.ri.struct.instantiation.IStructInstantiation;

public abstract class LinearStructModel<X, Y> implements IStructModel<X, Y> {

	IStructInstantiation<X, Y> instance;
	double[] parameters;
	
	public LinearStructModel(int dimpsi) {
		double[] w = new double[dimpsi];
		Arrays.fill(w, 0);
		this.setParameters(w);
	}
	
	public IStructInstantiation<X, Y> instantiation() {
		return instance;
	}

	public double[] getParameters() {
		return parameters;
	}

	public void setParameters(double[] w) {
		this.parameters = w;
	}

}
