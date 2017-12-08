package upmc.ri.struct.instantiation;

import java.util.Set;

import java.util.List;

import upmc.ri.struct.ranking.RankingOutput;
import upmc.ri.struct.ranking.RankingFunctions;;

public class RankingInstanciation implements IStructInstantiation<List<double[]>, RankingOutput> {

	public double[] psi(List<double[]> x, RankingOutput y) {
		// TODO Auto-generated method stub
		return null;
	}

	public double delta(RankingOutput y1, RankingOutput y2) {
		return 1 - RankingFunctions.averagePrecision(y2);
	}

	public Set<RankingOutput> enumerateY() {
		return null;
	}

}
