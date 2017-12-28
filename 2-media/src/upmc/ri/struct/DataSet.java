package upmc.ri.struct;

import java.io.Serializable;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class DataSet<X,Y>  implements Serializable{	
	/**
	 * 
	 */
	private static final long serialVersionUID = -3417522594699229035L;

	public List<STrainingSample<X,Y>> listTrain;
	public List<STrainingSample<X,Y>> listTest;
	
	public DataSet(List<STrainingSample<X, Y>> listTrain,List<STrainingSample<X, Y>> listTest) {
		super();
		this.listTrain = listTrain;
		this.listTest = listTest;
	}
	
	public Set<Y> outputs(){
		Set<Y> out= new LinkedHashSet<Y>();
		for(STrainingSample<X,Y> st : listTrain){
			out.add(st.output);
		}
		return out;
	}

	public List<STrainingSample<X, Y>> getTrain() {
		return this.listTrain;
	}
	
	public List<STrainingSample<X, Y>> getTest() {
		return this.listTest;
	}
	
	/** Gives the distribution of outputs in a list of training samples
	 * @param listTs The list to be analyzed
	 * @return A map linking each output to its number of appearances.
	 */
	private Map<Y, Integer> countLabels(List<STrainingSample<X, Y>> listTs) {
		Map<Y, Integer> count = new HashMap<Y, Integer>();
		for(STrainingSample<X, Y> ts: listTs){
			int freq = count.containsKey(ts.output) ? count.get(ts.output) : 0;
			count.put(ts.output, freq+1);
		}
		return count;
	}
	
	public Map<Y, Integer> countTrainLabels() {
		return this.countLabels(listTrain);
	}
	public Map<Y, Integer> countTestLabels() {
		return this.countLabels(listTest);
	}
}
