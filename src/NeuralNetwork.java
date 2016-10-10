import org.la4j.Matrix;
import org.la4j.Vector;

public class NeuralNetwork {
	
	private int numberOfLayers;
	private int[] layers;
	private Matrix[] weights;
	private Matrix[] biases;
	private Matrix[] activations;
	private Matrix training_inputs;
	private Matrix training_outputs;
	private Matrix test_inputs;
	private Matrix test_outputs;
	
	public NeuralNetwork(int[] layers,Matrix x,Matrix y,Matrix test_x,Matrix test_y){
		this.numberOfLayers = layers.length;
		this.setLayers(layers);
		this.weights = new Matrix[numberOfLayers-1];
		this.biases = new Matrix[numberOfLayers-1];
		this.setTraining_inputs(x);
		this.setTraining_outputs(rawValuesToVector(y, layers[numberOfLayers-1]));
		this.test_inputs = test_x;
		this.test_outputs = rawValuesToVector(test_y, layers[numberOfLayers-1]);
		for (int i = 0; i < numberOfLayers -1 ; i++) {
			this.weights[i] = Matrix.zero(layers[i+1], layers[i]);
			this.biases[i] = Matrix.zero(1, layers[i+1]);
			randomInit(this.weights[i]);
			randomInit(this.biases[i]);
			//printDimensions(this.weights[i]);
		}

		sgd(30,10,0.1);
		
	}
	
	public void feedforward(Matrix x){
		
		activations = new Matrix[layers.length];// first layer is the inputs
		activations[0] = x;
		//create arrays for biases x*getBiases()[j].columns() dimensions to be able to add them
		Matrix[] bias = new Matrix[layers.length-1];
		for(int j =0;j<layers.length-1;j++){
			bias[j] = Matrix.zero(x.rows(),getBiases()[j].columns());
			for(int i =0;i<x.rows();i++){
				bias[j].setRow(i, getBiases()[j].getRow(0));
			}
		}

		for (int i = 1; i < numberOfLayers ; i++) {
			//every activation array has a number of inputs as dimension. the other dimension matches the number neurons in each layer 
			getActivations()[i] = (getActivations()[i-1].multiply(getWeights()[i-1].transpose())).add(bias[i-1]) ;//W*a b
			getActivations()[i] = sigmoid(getActivations()[i]);//sigmoid(W*a)
			//printDimensions(getActivations()[i]);
		}
	}
	
	public void gd(int epochs,int mini_batch_size,double eta/*.Matrix train_x,Matrix train_y*/){
		
		System.out.println(evaluate()+" /"+test_inputs.rows());
		for (int e = 0; e < epochs; e++) {
			
			Matrix[][] accumulators = new Matrix[2][layers.length-1];
			for (int j = 0; j < layers.length-1; j++) {
				accumulators[0][j] =Matrix.zero(layers[j+1], layers[j]);
				accumulators[1][j] =Matrix.zero(1, layers[j+1]);
				
			}
			
			System.out.println("epoch: "+(e+1));

			for(int i=0;i<getTraining_inputs().rows();i++){
				Matrix[][] grads = backpropagation(getTraining_inputs().getRow(i).toRowMatrix(),getTraining_outputs().getRow(i).toRowMatrix());
				for (int j = 0; j < layers.length-1; j++) {
					accumulators[0][j] = accumulators[0][j].add(grads[0][j].transpose());
					accumulators[1][j] = accumulators[1][j].add(grads[1][j]);
				}
			}
			//System.out.println(accumulators[0][0].get(0, 0));
			
			for (int j = 0; j < layers.length-1; j++) {
				accumulators[0][j] = accumulators[0][j].multiply((double)1/getTraining_inputs().rows());
				accumulators[1][j] = accumulators[1][j].multiply((double)1/getTraining_inputs().rows());
			}
			//System.out.println(accumulators[0][0].get(0, 0));
			
			for (int i = 0; i <layers.length-1; i++) {
				weights[i] = weights[i].subtract(accumulators[0][i].multiply(eta));
				biases[i] = biases[i].subtract(accumulators[1][i].multiply(eta));
			}
			System.out.println(evaluate()+" /"+test_inputs.rows());
		}
		
	}
	
	public void sgd(int epochs,int mini_batch_size,double eta/*.Matrix train_x,Matrix train_y*/){
		
		
		System.out.println(evaluate()+" /"+test_inputs.rows());
		
		for (int e = 0; e < epochs; e++) {
			
			Matrix[][] accumulators = new Matrix[2][layers.length-1];
			System.out.println("epoch: "+(e+1));
			//feedforward(getTraining_inputs().copy());
			//costFunction();
			
			int batches = getTraining_inputs().rows() / mini_batch_size;
			//System.out.println(batches);
			for (int b = 0; b < batches; b++) {
				for (int j = 0; j < layers.length-1; j++) {
					accumulators[0][j] =Matrix.zero(layers[j+1], layers[j]);
					accumulators[1][j] =Matrix.zero(1, layers[j+1]);
					
				}	
				
				for(int i=mini_batch_size*b;i<mini_batch_size*(b+1);i++){
					Matrix[][] grads = backpropagation(getTraining_inputs().getRow(i).toRowMatrix(),getTraining_outputs().getRow(i).toRowMatrix());
					for (int j = 0; j < layers.length-1; j++) {
						accumulators[0][j] = accumulators[0][j].add(grads[0][j].transpose());
						accumulators[1][j] = accumulators[1][j].add(grads[1][j]);
					}
				}
				
				for (int j = 0; j < layers.length-1; j++) {
					accumulators[0][j] = accumulators[0][j].multiply((double)1/mini_batch_size);
					accumulators[1][j] = accumulators[1][j].multiply((double)1/mini_batch_size);
				}
				
				for (int i = 0; i <layers.length-1; i++) {
					weights[i] = weights[i].subtract(accumulators[0][i].multiply(eta));
					biases[i] = biases[i].subtract(accumulators[1][i].multiply(eta));
				}
				
			}
			System.out.println(evaluate()+" /"+test_inputs.rows());

		}
		
	}
	
	public int evaluate(){
		feedforward(test_inputs);
		int corrects = 0;
		for (int j = 0; j < getActivations()[getNumberOfLayers()-1].rows(); j++) {
			
		
			Vector v = getActivations()[getNumberOfLayers()-1].getRow(j);
			int pos = maxPosition(v);
			if(test_outputs.get(j, pos)==1){
				corrects++;
			}
		
		}
		return(corrects);
	}
	
	private int maxPosition(Vector v){
		int position = 0;
		double max = 0.0;
		for (int i = 0; i < v.length(); i++) {
			if (v.get(i)>max){
				max = v.get(i);
				position = i;
			}
		}
		return position;
	}
	
	
	
	public Matrix[][] backpropagation(Matrix x, Matrix y){
		//feed forward with one trainining set		
		Matrix[][] gradients = new Matrix[2][layers.length-1];// for weights and bias
		Matrix[] a = new Matrix[layers.length];
		Matrix[] z = new Matrix[layers.length];//z[0] remains unitilized
		Matrix[] delta = new Matrix[layers.length]; //there is no error in layer 0
		//System.out.println("activations");
		a[0] = x;
		//printDimensions(a[0]);
		for (int i = 1; i < numberOfLayers ; i++) {
			//every activation array has a number of inputs as dimension. the other dimension matches the number neurons in each layer 
			z[i] = (a[i-1].multiply(getWeights()[i-1].transpose())).add(getBiases()[i-1]);  //W*a +b
			a[i] = sigmoid(z[i]);//sigmoid(z)

			//printDimensions(a[i]);
		}
		
		//find deltas
		//System.out.println("delta");
		delta[getNumberOfLayers()-1] = a[getNumberOfLayers()-1].subtract(y);
		//printDimensions(delta[getNumberOfLayers()-1]);
		for (int i = numberOfLayers-2; i >=1 ; i--) {
			//every activation array has a number of inputs as dimension. the other dimension matches the number neurons in each layer (need to add one for the bias)
			delta[i] = (delta[i+1].multiply( getWeights()[i])).hadamardProduct(sigmoidPrime(z[i]));//delta[i] = W[i]*delta(i+1).*sigmoidPrime(z[i])
			//printDimensions(delta[i]);

		}
		//System.out.println("gradients");
		for (int i = 0; i< getNumberOfLayers()-1 ; i++) {
			gradients[0][i] = a[i].transpose().multiply(delta[i+1]);// wGrad = a(i)*delta(i+1)
			gradients[1][i] = delta[i+1];
			//printDimensions(Wgradients[i]);

		}
		
		return gradients;
		
		//printDimensions(y);
		//printDimensions(delta[getNumberOfLayers()-1]);
		
	}
	
	public void costFunction(){
		// least square cost
		Matrix  m = getActivations()[getNumberOfLayers()-1].subtract(getTraining_outputs()); //a-y
		m = m.hadamardProduct(m);//(a-y)^2
		double sum=0.0;
		for (int i = 0; i < m.rows(); i++) {
			sum= sum +m.getRow(i).sum();
		}
		System.out.println("Error:"+ sum/(getTraining_inputs().rows()*2));//1/2n * sum((a-y)^2)
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
		
		Matrix prime = sigmoid(z).hadamardProduct(Matrix.constant(z.rows(), z.columns(), 1.0).subtract(sigmoid(z)));
		return prime;
	}
	
	
	private Matrix rawValuesToVector(Matrix m, int sizeVector){
		Matrix y = Matrix.zero(m.rows(), sizeVector) ;
		for (int i = 0; i < m.rows(); i++) {
			y.set(i,(int) m.get(i,0),1.0);
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
	
/*	private static Matrix  addBias(Matrix m){
		m = m.insertColumn(0, Vector.constant(m.rows(), 1.0));
		return m;
	}*/
	
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

	public Matrix[] getBiases() {
		return biases;
	}

	public void setBiases(Matrix[] biases) {
		this.biases = biases;
	}
	


}
