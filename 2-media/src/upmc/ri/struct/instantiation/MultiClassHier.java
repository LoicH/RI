package upmc.ri.struct.instantiation;

import edu.cmu.lti.lexical_db.NictWordNet;
import edu.cmu.lti.ws4j.RelatednessCalculator;
import edu.cmu.lti.ws4j.impl.WuPalmer;

public class MultiClassHier extends MultiClass{
	double [][] distances;
	
	public MultiClassHier() {
		super();
		RelatednessCalculator rc = new WuPalmer(new NictWordNet());
		this.distances = new double[this.getSet().size()][this.getSet().size()];
		int index_line;
		int index_column;
		for(String y1: this.getSet()) {
			index_line = this.getMap().get(y1);
			for (String y2 : this.getSet()) {
				index_column = this.getMap().get(y2);
				if (index_line > index_column && !y1.equals(y2)) {
					this.distances[index_line][index_column] = rc.calcRelatednessOfWords(y1, y2);
					this.distances[index_column][index_line] = this.distances[index_line][index_column];
				}
			} 
		}
	}
	
	public double delta(String y1, String y2) {
		double result;
		if (!y1.equals(y2)){
			result = 1 - this.distances[this.getMap().get(y1)][this.getMap().get(y2)];
		}
		else result = 0;
		return result;
	}
}
