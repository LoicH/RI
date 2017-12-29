package upmc.ri.utils;

public class MatrixOperations {
	/** Transforms the matrix such that the sum of elements in a column equals 1
	 * @param matrix
	 * @return The column-stochastic matrix
	 */
	public static double[][] normalizeCol(double[][] matrix){
		double[][] normMat = matrix.clone();
		// Iterate over all columns
		for(int j=0;j<matrix[0].length;j++){
			// Compute the sum of elements in the column
			double colSum = 0;
			for (int i = 0; i<matrix.length; i++){
				colSum += matrix[i][j];
			}
			// Divide the whole column by its sum
			if (colSum != 0){
				for (int i = 0; i<matrix.length; i++){
					normMat[i][j] /= colSum;
				}
			}
		}
		return normMat;
	}

}
