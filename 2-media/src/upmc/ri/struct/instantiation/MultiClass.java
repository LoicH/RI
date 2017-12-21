package upmc.ri.struct.instantiation;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.MatrixVisualization;
import upmc.ri.io.ImageNetParser;

public class MultiClass implements IStructInstantiation<double[], String> {
	
	
	private int dim;
	private Set<String> set;
	private Map<String,Integer> map;
	

	public MultiClass() {
		this.dim = 250;
		this.set = ImageNetParser.classesImageNet();
		this.map = new HashMap<>();
		
		int indexY = 0;
		for (String y: this.set) {
			this.map.put(y, indexY);
			indexY++;
		
		}
	}
	
	public double[] psi(double[] x,String y) {
		double[] psiVal = new double[this.dim * this.set.size()];
		Arrays.fill(psiVal, 0);
		int index;
		try {
			index = this.map.get(y);

			int count = 0;
			for (int i = index * this.dim; i < (index+1) * this.dim; i++) {
				psiVal[i] = x[count];
				count++;
			}
		}
		catch (NullPointerException e) {
			System.out.println("Null pointer, y:"+y);
			for(Map.Entry<String, Integer> entry: this.map.entrySet()) {
				System.out.println(entry.getKey() + ":" + entry.getValue());
			}
		}
		return psiVal;
	}
	
	public double delta(String y1,String y2) {
		int result = 0;
		if (!y1.equals(y2)){
			result = 1;
		}
		return result;
	}
	
	public Set<String> enumerateY(){
		return this.set;
	}
	
	/** Computes and shows the confusion matrix
	 * @param predictions The list of labels predicted for the test set
	 * @param gt The list of true labels
	 * @return Nothing
	 */
	public double [][] confusionMatrix(List<String> predictions, List<String> gt) {
		double[][] matrix = new double[this.set.size()][this.set.size()];
		for(int i = 0; i<matrix.length; i++) {
			Arrays.fill(matrix[i], 0);
		}
		int index_line; // predicted
		int index_column; // real value
		for(int i = 0; i < predictions.size(); i++) {
			index_line = this.map.get(gt.get(i));
			index_column = this.map.get(predictions.get(i));
			matrix[index_line][index_column]++;
		}
		// TODO Visualizing Confusion Matrix with EJLM.MatrixVisualization
		DenseMatrix64F confMatrix = new DenseMatrix64F(matrix);
		MatrixVisualization.show(confMatrix, "Matrice de confusion");
		return matrix;
		
	}

	public Set<String> getSet() {
		return this.set;
	}

	public int getDim() {
		return dim;
	}

	public void setDim(int dim) {
		this.dim = dim;
	}

	public Map<String, Integer> getMap() {
		return map;
	}

	public void setMap(Map<String, Integer> map) {
		this.map = map;
	}
}

