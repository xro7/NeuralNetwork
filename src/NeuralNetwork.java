import org.la4j.Matrix;
import org.la4j.Vector;
import org.la4j.matrix.functor.MatrixProcedure;

public class NeuralNetwork {
	
	private int numberOfLayers;
	private int[] layers;
	private Matrix[] weights;
	private Matrix[] activations;
	private Matrix training_inputs;
	private Matrix training_outputs;
	
	public NeuralNetwork(int[] layers,Matrix x,Matrix y){
		this.numberOfLayers = layers.length;
		this.setLayers(layers);
		this.weights = new Matrix[numberOfLayers-1];
		this.setTraining_inputs(x);
		this.setTraining_outputs(rawValuesToVector(y, layers[numberOfLayers-1]));
		for (int i = 0; i < numberOfLayers -1 ; i++) {
			this.weights[i] = Matrix.zero(layers[i+1], layers[i] + 1 /*(this is the bias term that added to the weights)*/);
			randomInit(this.weights[i]);
			//printDimensions(this.weights[i]);
		}
		feedforward();
		costFunction();
		backpropagation(getTraining_inputs().getRow(0).toRowMatrix(),getTraining_outputs().getRow(0).toRowMatrix());
		
	}
	
	public void feedforward(){
		
		activations = new Matrix[layers.length];// first layer is the inputs
		activations[0] = getTraining_inputs().copy();
		
		for (int i = 1; i < numberOfLayers ; i++) {
			//every activation array has a number of inputs as dimension. the other dimension matches the number neurons in each layer (need to add one for the bias)
			getActivations()[i] = getActivations()[i-1].multiply(getWeights()[i-1].transpose());  //W*a
			if (i != numberOfLayers-1){
				getActivations()[i] = addBias(getActivations()[i]); //add bias column
			}
			sigmoid(getActivations()[i]);//sigmoid(W*a)
			//printDimensions(getActivations()[i]);
		}
	}
	
	public void backpropagation(Matrix x, Matrix y){
		//feed forward with one trainining set		
		Matrix[] gradients = new Matrix[layers.length-1];
		Matrix[] a = new Matrix[layers.length];
		Matrix[] z = new Matrix[layers.length];//z[0] remains unitilized
		Matrix[] delta = new Matrix[layers.length]; //there is no error in layer 0
		System.out.println("activations");
		a[0] = x;
		printDimensions(a[0]);
		for (int i = 1; i < numberOfLayers ; i++) {
			//every activation array has a number of inputs as dimension. the other dimension matches the number neurons in each layer (need to add one for the bias)
			z[i] = a[i-1].multiply(getWeights()[i-1].transpose());  //W*a
			if (i != numberOfLayers-1){
				z[i] = addBias(z[i]); //add bias column
			}
			a[i] = sigmoid(z[i]);//sigmoid(W*a)

			printDimensions(a[i]);
		}
		
		//find deltas
		System.out.println("delta");
		delta[getNumberOfLayers()-1] = a[getNumberOfLayers()-1].subtract(y);
		printDimensions(delta[getNumberOfLayers()-1]);
		for (int i = numberOfLayers-2; i >=1 ; i--) {
			//every activation array has a number of inputs as dimension. the other dimension matches the number neurons in each layer (need to add one for the bias)
			delta[i] = (delta[i+1].multiply( getWeights()[i])).hadamardProduct(sigmoidPrime(z[i]));
			printDimensions(delta[i]);

		}
		
		for (int i = 0; i< getNumberOfLayers()-1 ; i++) {
			gradients[i] = a[i].transpose().multiply(delta[i+1]);
			printDimensions(gradients[i]);

		}
		
		//printDimensions(y);
		//printDimensions(delta[getNumberOfLayers()-1]);
		
	}
	
	public void costFunction(){
		// least square cost
		Matrix  m = getActivations()[getNumberOfLayers()-1].subtract(getTraining_outputs()); //a-y
		m.each(new MatrixProcedure() {
			
			@Override
			public void apply(int arg0, int arg1, double arg2) {
				m.set(arg0, arg1,arg2*arg2);
			}
		});//(a-y)^2
		System.out.println("Error:"+ m.sum()/(getTraining_inputs().rows()*2));//1/2n * sum((a-y)^2)
		//printDimensions(m);
	}
	
	private static void randomInit(Matrix matrix){
		java.util.Random r = new java.util.Random();
		for (int i = 0; i < matrix.rows(); i++) {
			for (int j = 0; j < matrix.columns(); j++) {
				matrix.set(i, j, r.nextGaussian());
			}
		}
	}
	
	private Matrix sigmoid(Matrix t){
		Matrix sig =  Matrix.constant(t.rows(), t.columns(), 0.0);
		for (int i = 0; i < t.rows(); i++) {
			for (int j = 0; j < t.columns(); j++) {
				sig.set(i, j, 1.0 / (1.0 + Math.exp(-t.get(i,j))));
			}
		}
		return(sig);
		
	}
	
	private Matrix sigmoidPrime(Matrix z){
		
		Matrix prime = sigmoid(z).hadamardProduct(sigmoid(Matrix.constant(z.rows(), z.columns(), 1.0).subtract(z)));
		return prime;
	}
	
	
	private Matrix rawValuesToVector(Matrix m, int sizeVector){
		Matrix y = Matrix.zero(m.rows(), sizeVector) ;
		for (int i = 0; i < m.rows(); i++) {
			y.set(i,(int) m.get(i,0)-1,1.0);
		}
		return y;
	}

	public int[] getLayers() {
		return layers;
	}

	public void setLayers(int[] layers) {
		this.layers = layers;
	}

	public int getNumberOfLayers() {
		return numberOfLayers;
	}

	public void setNumberOfLayers(int numberOfLayers) {
		this.numberOfLayers = numberOfLayers;
	}

	public Matrix[] getWeights() {
		return weights;
	}

	public void setWeights(Matrix[] weights) {
		this.weights = weights;
	}
	
	public Matrix[] getActivations() {
		return activations;
	}


	public void setActivations(Matrix[] activations) {
		this.activations = activations;
	}
	
	public static void printDimensions(Matrix m){
		System.out.println(m);
		System.out.println("Diamensions of matrix are "+m.rows()+"X"+m.columns() );
	}
	
	private static Matrix  addBias(Matrix m){
		m = m.insertColumn(0, Vector.constant(m.rows(), 1.0));
		return m;
	}
	
	public Matrix getTraining_inputs() {
		return training_inputs;
	}

	public void setTraining_inputs(Matrix training_inputs) {
		this.training_inputs = training_inputs;
	}

	public Matrix getTraining_outputs() {
		return training_outputs;
	}

	public void setTraining_outputs(Matrix training_outputs) {
		this.training_outputs = training_outputs;
	}
	


}
