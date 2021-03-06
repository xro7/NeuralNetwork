
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;
import org.la4j.Matrix;
import org.la4j.Vector;
import org.la4j.matrix.DenseMatrix;


public class Main {

	public static void main(String[] args) throws FileNotFoundException {

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
		int train_size = 60000;
		x = DenseMatrix.zero(train_size , 784);
		for (int i = 0; i < train_size ; i++) {
			Vector v = Vector.fromArray(images.get(i).getData());
			x.setRow(i, v);
		}
		
		y = DenseMatrix.zero(train_size , 1);
		for (int i = 0; i < train_size ; i++) {
			y.setRow(i, images.get(i).getLabel());
		}
		
		DenseMatrix set = (DenseMatrix) x.insertColumn(0, y.getColumn(0));
		DenseMatrix trainingSet = (DenseMatrix) set.sliceTopLeft(50000, set.columns());
		DenseMatrix validationSet = (DenseMatrix) set.sliceBottomRight(50000, 0);
		printDimensions(trainingSet);
		printDimensions(validationSet);
		
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
		
		DenseMatrix testSet = (DenseMatrix) test_x.insertColumn(0, test_y.getColumn(0));
		printDimensions(testSet);
		
		double elapsedTime = System.nanoTime() - start;
		
		System.out.println("Done reading images after "+ elapsedTime/1000000000+" seconds");
		
		
/*		
		ProcessInputs pi = new ProcessInputs(7767,561,"data/train/X_train.txt");
		ProcessInputs pi2 = new ProcessInputs(7767,561,"data/train/Y_train.txt");
		ProcessInputs pi3 = new ProcessInputs(3162,561,"data/test/X_test.txt");
		ProcessInputs pi4 = new ProcessInputs(3162,561,"data/test/Y_test.txt");
		DenseMatrix trainingSet = (DenseMatrix) pi.getInputs().insertColumn(0, pi2.getInputs().getColumn(0));
		DenseMatrix testSet = (DenseMatrix) pi3.getInputs().insertColumn(0, pi4.getInputs().getColumn(0));
	*/
		
		NeuralNetwork nn = new NeuralNetwork(new int[]{784,100,50,10},trainingSet,validationSet,testSet);
		nn.sgd(70,10,0.01,nn.new CrossEntropyCostFunction(10),nn.new Sigmoid(),false);
		//NeuralNetwork nn = new NeuralNetwork(new int[]{561,60,12},trainingSet,(DenseMatrix) Matrix.zero(4, 4),testSet);
		
	}
	
	
	public static void printDimensions(Matrix m){
		//System.out.println(m);
		System.out.println("Diamensions of matrix are "+m.rows()+"X"+m.columns() );
	}
	
	

}
