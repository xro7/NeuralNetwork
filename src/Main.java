import org.la4j.Vector;

import org.la4j.Matrix;
import org.la4j.matrix.DenseMatrix;
import org.la4j.matrix.dense.Basic1DMatrix;
import org.la4j.matrix.dense.Basic2DMatrix;
import org.la4j.matrix.functor.MatrixProcedure;
import org.la4j.vector.DenseVector;
import org.la4j.vector.dense.BasicVector;

public class Main {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
/*		Matrix x = Matrix.from2DArray(new double[][]{{1,2},{1,2},{1,2}});
		System.out.println(x);
		x = x.insertColumn(0, Vector.fromArray(new double[]{6,6,5}));
		System.out.println(x);*/
		

		//ProcessInputs pi = new ProcessInputs(7767,561,"data/train/X_train.txt");
		ProcessInputs pi = new ProcessInputs(30,12,"data/train/X_train.txt");
		ProcessInputs pi2 = new ProcessInputs(30,1,"data/train/Y_train.txt");
		Matrix x = pi.getInputs();
		Matrix y_raw = pi2.getInputs();
		//Matrix y = rawValuesToVector(y_raw,12);

		//printDimensions(y_raw);
		x = addBias(x);
		//printDimensions(x);
	
/*		ProcessInputs pi2 = new ProcessInputs(7767,561,"data/train/Y_train.txt");
		ProcessInputs pi3 = new ProcessInputs(3162,561,"data/test/X_test.txt");
		ProcessInputs pi4 = new ProcessInputs(3162,561,"data/test/Y_test.txt");*/
		//NeuralNetwork nn = new NeuralNetwork(new int[]{561,20,1});
		NeuralNetwork nn = new NeuralNetwork(new int[]{12,4,12},x,y_raw);
		//printDimensions(nn.getWeights()[0]);
		
		
		
		
		
		
	}
	
	private static Matrix rawValuesToVector(Matrix m, int sizeVector){
		Matrix y = Matrix.zero(m.rows(), sizeVector) ;
		for (int i = 0; i < m.rows(); i++) {
			y.set(i,(int) m.get(i,0)-1,1.0);
		}
		return y;
	}
	
	public static void printDimensions(Matrix m){
		System.out.println(m);
		System.out.println("Diamensions of matrix are "+m.rows()+"X"+m.columns() );
	}
	
	private static Matrix  addBias(Matrix m){
		m = m.insertColumn(0, Vector.constant(m.rows(), 1.0));
		return m;
	}
	


}
