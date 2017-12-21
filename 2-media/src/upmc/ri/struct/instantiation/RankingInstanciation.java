package upmc.ri.struct.instantiation;

import java.util.List;
import java.util.Set;

import upmc.ri.struct.ranking.RankingFunctions;
import upmc.ri.struct.ranking.RankingOutput;
import upmc.ri.utils.VectorOperations;;

public class RankingInstanciation implements IStructInstantiation<List<double[]>, RankingOutput> {

	private int dim;
	
	public RankingInstanciation() {
		this.dim = 250;
	}
	/**
	 * @param x List of double array that are image representation (BoW in d dimension)
	 * @param y RankingOuput object which contains a List of ordered images of the input x according to a request for ranking purpose
	 * @return The feature map for the specific ranking problem (ie binary classification were class 1 is all wanted items and class B all others items)
	 */
	public double[] psi(List<double[]> x, RankingOutput y) {
		
		double[] psi = new double[x.get(1).length];
		double[] temporaryPsi;
		try {
			for(int i: y.getPlus()){
				for(int j: y.getMinus()){
					temporaryPsi = VectorOperations.scalarProduct(
							VectorOperations.substract(x.get(i),x.get(j))
							, y.isBefore(i, j)); 
					psi = VectorOperations.add(psi,temporaryPsi);
				}
			}
		} catch(NullPointerException e) {}
		
		return psi;
	}

	/**
	 * @param y1 and y2 Lists of ordered images of the input x according to a request for ranking purpose
	  * @return The error of our ranking based of the Average Precision (Area under the Precision Recall curve)
	 */
	public double delta(RankingOutput y1, RankingOutput y2) {
		return 1 - RankingFunctions.averagePrecision(y2);
		// TODO see if it ain't averagePrecision(y1) that must be called
	}
	
	/**
		Info : can't be specified in our specific problem because the size of the output space is untractable exponential number of solutions
	  */
	public Set<RankingOutput> enumerateY() {
		return null;
	}
	
	public int getDim() {
		return dim;
	}
	public void setDim(int dim) {
		this.dim = dim;
	}

}
