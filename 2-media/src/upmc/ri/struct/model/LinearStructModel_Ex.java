package upmc.ri.struct.model;

import upmc.ri.struct.STrainingSample;
import upmc.ri.utils.VectorOperations;

public class LinearStructModel_Ex<X, Y> extends LinearStructModel<X, Y> {

	public LinearStructModel_Ex(int dimpsi) {
		super(dimpsi);
	}
	

	public Y predict(X x) {
		double[] w = super.getParameters();
		Y yMax = null;
		double valMax = Double.MIN_VALUE;
		for (Y y : this.instance.enumerateY()){
			double val = VectorOperations.dot(w, super.instantiation().psi(x, y));
			if (val > valMax){
				valMax = val;
				yMax = y;
			}
		}
		return yMax;
	}

	public Y lai(STrainingSample<X, Y> ts) {
		double[] w = super.getParameters();
		Y yMax = null;
		double valMax = Double.MIN_VALUE;
		for (Y y : this.instance.enumerateY()){
			double val = VectorOperations.dot(w, super.instantiation().psi(ts.input, y));
			val += super.instantiation().delta(y, ts.output);
			if (val > valMax){
				valMax = val;
				yMax = y;
			}
		}
		return yMax;
	}



}
