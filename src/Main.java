
import java.io.IOException;
import java.util.List;

import org.la4j.Matrix;
import org.la4j.Vector;

public class Main {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
/*		Matrix x = Matrix.from2DArray(new double[][]{{1,2},{1,2},{1,2}});
		System.out.println(x);
		x = x.insertColumn(0, Vector.fromArray(new double[]{6,6,5}));
		System.out.println(x);*/
		List<DigitImage> images =null;
		Mnist m = new Mnist("data/train-labels.idx1-ubyte","data/train-images.idx3-ubyte");
		try {
			images = m.loadDigitImages();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		int size = 5000;
		Matrix x = Matrix.zero(size, 784);
		for (int i = 0; i < size; i++) {
			Vector v = Vector.fromArray(images.get(i).getData());
			x.setRow(i, v);
		}
		
		Matrix y = Matrix.zero(size, 1);
		for (int i = 0; i < size; i++) {
			images.get(i).getData();
			y.setRow(i, images.get(i).getLabel());
		}
		
/*		//ProcessInputs pi = new ProcessInputs(7767,561,"data/train/X_train.txt");
		ProcessInputs pi = new ProcessInputs(5000,561,"data/train/X_train.txt");
		ProcessInputs pi2 = new ProcessInputs(5000,1,"data/train/Y_train.txt");
		Matrix x = pi.getInputs();
		Matrix y_raw = pi2.getInputs();
		//Matrix y = rawValuesToVector(y_raw,12);
*/
		//printDimensions(y_raw);
		//x = addBias(x);
		//printDimensions(x);
	
/*		ProcessInputs pi2 = new ProcessInputs(7767,561,"data/train/Y_train.txt");
		ProcessInputs pi3 = new ProcessInputs(3162,561,"data/test/X_test.txt");
		ProcessInputs pi4 = new ProcessInputs(3162,561,"data/test/Y_test.txt");*/
		//NeuralNetwork nn = new NeuralNetwork(new int[]{561,20,1});
		NeuralNetwork nn = new NeuralNetwork(new int[]{784,100,10},x,y);
		//printDimensions(nn.getWeights()[0]);
		
		
		
		//printDimensions(sigmoidPrime(Matrix.constant(3, 4, 3)));
		
		
	}
	
	
	public static void printDimensions(Matrix m){
		System.out.println(m);
		System.out.println("Diamensions of matrix are "+m.rows()+"X"+m.columns() );
	}
	
/*	private static Matrix  addBias(Matrix m){
		m = m.insertColumn(0, Vector.constant(m.rows(), 1.0));
		return m;
	}*/
	
/*	private static Matrix sigmoidPrime(Matrix z){
		Matrix prime = z.hadamardProduct((z));
		return prime;
	}
	private static Matrix sigmoid(Matrix t){
		Matrix sig =  Matrix.constant(t.rows(), t.columns(), 0.0);
		for (int i = 0; i < t.rows(); i++) {
			for (int j = 0; j < t.columns(); j++) {
				sig.set(i, j, 1.0 / (1.0 + Math.exp(-t.get(i,j))));
			}
		}
		return(sig);
		
	}*/
	


}
