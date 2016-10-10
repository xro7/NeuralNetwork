
import java.io.IOException;
import java.util.List;

import org.la4j.Matrix;
import org.la4j.Vector;

public class Main {

	public static void main(String[] args) {
		
/*		Matrix x = Matrix.from2DArray(new double[][]{{1,2},{1,2},{1,2}});
		System.out.println(x);
		x = x.insertColumn(0, Vector.fromArray(new double[]{6,6,5}));
		System.out.println(x);*/
		List<DigitImage> images =null;
		List<DigitImage> images2 =null;
		Mnist m = new Mnist("data/train-labels.idx1-ubyte","data/train-images.idx3-ubyte");
		Mnist m2 = new Mnist("data/t10k-labels.idx1-ubyte","data/t10k-images.idx3-ubyte");
		
		try {
			images = m.loadDigitImages();
		} catch (IOException e) {
			e.printStackTrace();
		}
		int train_size = 20000;
		Matrix x = Matrix.zero(train_size , 784);
		for (int i = 0; i < train_size ; i++) {
			Vector v = Vector.fromArray(images.get(i).getData());
			x.setRow(i, v);
		}
		
		Matrix y = Matrix.zero(train_size , 1);
		for (int i = 0; i < train_size ; i++) {
			y.setRow(i, images.get(i).getLabel());
		}
		
		try {
			images2 = m2.loadDigitImages();
		} catch (IOException e) {
			e.printStackTrace();
		}
		int test_size = 1000;
		Matrix test_x = Matrix.zero(test_size , 784);
		for (int i = 0; i < test_size ; i++) {
			Vector v = Vector.fromArray(images2.get(i).getData());
			test_x.setRow(i, v);
		}
		
		Matrix test_y = Matrix.zero(test_size , 1);
		for (int i = 0; i < test_size ; i++) {
			test_y.setRow(i, images2.get(i).getLabel());
		}
		System.out.println("Done reading images");
/*		//ProcessInputs pi = new ProcessInputs(7767,561,"data/train/X_train.txt");
		ProcessInputs pi = new ProcessInputs(5000,561,"data/train/X_train.txt");
		ProcessInputs pi2 = new ProcessInputs(5000,1,"data/train/Y_train.txt");
		Matrix x = pi.getInputs();
		Matrix y_raw = pi2.getInputs();
*/
		
		NeuralNetwork nn = new NeuralNetwork(new int[]{784,100,10},x,y,test_x,test_y);

		
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
