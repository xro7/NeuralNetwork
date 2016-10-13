
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;
import org.la4j.Matrix;
import org.la4j.Vector;
import org.la4j.matrix.DenseMatrix;


public class Main {

	public static void main(String[] args) throws FileNotFoundException {
		
/*		Matrix x = Matrix.from2DArray(new double[][]{{1,2},{1,2},{1,2}});
		System.out.println(x);
		x = x.insertColumn(0, Vector.fromArray(new double[]{6,6,5}));
		System.out.println(x);*/
			
		DenseMatrix x;
		DenseMatrix y;
		DenseMatrix test_x;
		DenseMatrix test_y;
		
		boolean fromCsv =true;
		//if( !fromCsv){
		double start = System.nanoTime();  
		List<DigitImage> images =null;
		List<DigitImage> images2 =null;
		Mnist m = new Mnist("data/train-labels.idx1-ubyte","data/train-images.idx3-ubyte");
		Mnist m2 = new Mnist("data/t10k-labels.idx1-ubyte","data/t10k-images.idx3-ubyte");
		
		try {
			images = m.loadDigitImages();
		} catch (IOException e) {
			e.printStackTrace();
		}
		int train_size = 50000;
		x = DenseMatrix.zero(train_size , 784);
		for (int i = 0; i < train_size ; i++) {
			Vector v = Vector.fromArray(images.get(i).getData());
			x.setRow(i, v);
		}
		
		y = DenseMatrix.zero(train_size , 1);
		for (int i = 0; i < train_size ; i++) {
			y.setRow(i, images.get(i).getLabel());
		}
		
		try {
			images2 = m2.loadDigitImages();
		} catch (IOException e) {
			e.printStackTrace();
		}
		int test_size = 10000;
		test_x = DenseMatrix.zero(test_size , 784);
		for (int i = 0; i < test_size ; i++) {
			Vector v = Vector.fromArray(images2.get(i).getData());
			test_x.setRow(i, v);
		}
		
		test_y = DenseMatrix.zero(test_size , 1);
		for (int i = 0; i < test_size ; i++) {
			test_y.setRow(i, images2.get(i).getLabel());
		}
		double elapsedTime = System.nanoTime() - start;
		System.out.println("Done reading images after "+ elapsedTime/1000000000+" seconds");
	
		
		NeuralNetwork nn = new NeuralNetwork(new int[]{784,100,10},x,y,test_x,test_y);

		
	}
	
	
	public static void printDimensions(Matrix m){
		System.out.println(m);
		System.out.println("Diamensions of matrix are "+m.rows()+"X"+m.columns() );
	}
	
	

}
