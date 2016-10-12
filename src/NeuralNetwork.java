import org.la4j.Matrix;
import org.la4j.Vector;
import org.la4j.matrix.functor.MatrixFunction;
import org.la4j.matrix.functor.MatrixProcedure;

public class NeuralNetwork {
	
	private int numberOfLayers;
	private int[] layers;
	private Matrix[] weights;
	private Matrix[] biases;
	private Matrix[] activations;
	private Matrix[] zeta;
	private Matrix training_inputs;
	private Matrix training_outputs;
	private Matrix test_inputs;
	private Matrix test_outputs;
	
	public NeuralNetwork(int[] layers,Matrix x,Matrix y,Matrix test_x,Matrix test_y){
		this.numberOfLayers = layers.length;
		this.setLayers(layers);
		this.activations = new Matrix[layers.length];
		this.zeta = new Matrix[layers.length];
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

		sgd2(30,20,0.1,new CrossEntropyCostFunction(),false);
		
	
	}
	
	public double[] feedforward(Matrix x){
		
		activations[0] = x;		
		Matrix[] bias = new Matrix[layers.length-1];
		double start;
		double[] elapsedTime = new double[numberOfLayers-1];
		for (int i = 1; i < numberOfLayers ; i++) {
			start = System.nanoTime();
			Matrix unary = Matrix.unit(getActivations()[i-1].rows(), 1);
			bias[i-1] = unary.multiply(getBiases()[i-1]);//need to make bias[i] from 1 X layers[i] to getActivations()[i-1].rows() X layers[i] so as to add it to z
			zeta[i] = (getActivations()[i-1].multiply(getWeights()[i-1].transpose())).add(bias[i-1]) ;//z =W*a b			
			getActivations()[i] = sigmoid(getZeta()[i]);//sigmoid(z)
			elapsedTime[i-1] = System.nanoTime() - start;
		}
		
		return elapsedTime;
		
	}		
	
	public void sgd2(int epochs,int mini_batch_size,double eta,CostFunction cost,boolean printCost){
		
		if (printCost){
			feedforward(getTraining_inputs());
			System.out.println("Initial Cost is:" + cost.cost(getActivations()[getNumberOfLayers()-1], getTraining_outputs()));
		}
		
		System.out.println(evaluate()+" /"+test_inputs.rows());//evaluate test set with random weights
		int batches = getTraining_inputs().rows() / mini_batch_size;//number of batches		 
		Matrix[][] accumulators = new Matrix[2][layers.length-1];

		for (int e = 0; e < epochs; e++) {
			
			System.out.println("epoch: "+(e+1));
			double start = System.nanoTime();   
						
			for (int b = 0; b < batches; b++) {
				Matrix x = getTraining_inputs().slice(b*mini_batch_size, 0,(b+1)*mini_batch_size, getTraining_inputs().columns());
				Matrix y = getTraining_outputs().slice(b*mini_batch_size, 0,(b+1)*mini_batch_size, getTraining_outputs().columns());

				accumulators = backpropagationVersion2(x, y,cost);
	
				//divide gradients by 1/mini_batch_size
				for (int j = 0; j < layers.length-1; j++) {
					accumulators[0][j] = accumulators[0][j].multiply((double)1/mini_batch_size);
					accumulators[1][j] = accumulators[1][j].multiply((double)1/mini_batch_size);
				}
				
				//update weights and bias subtracting the gradients multiplied by learning rate
				for (int i = 0; i <layers.length-1; i++) {
					weights[i] = weights[i].subtract(accumulators[0][i].multiply(eta).transpose());
					biases[i] = biases[i].subtract(accumulators[1][i].multiply(eta));
				}			
			}
			
			if (printCost){
				feedforward(getTraining_inputs());
				System.out.println("Cost is:" + cost.cost(getActivations()[getNumberOfLayers()-1], getTraining_outputs()));
			}
			
			double elapsedTime = System.nanoTime() - start;
			System.out.println(elapsedTime/1000000000+" seconds for this epoch");
			System.out.println(evaluate()+" /"+test_inputs.rows());

		}
		//System.out.println(evaluate()+" /"+test_inputs.rows());//evaluate test set with random weights
		
	}
	
	public Matrix[][] backpropagationVersion2(Matrix x, Matrix y,CostFunction cost){
		//feed forward 	
		feedforward(x);
		
		Matrix[][] gradients = new Matrix[2][layers.length-1];// for weights and bias
		Matrix[] delta = new Matrix[layers.length];
		
		//find deltas
		//delta[getNumberOfLayers()-1] = getActivations()[getNumberOfLayers()-1].subtract(y).hadamardProduct(sigmoidPrime(getZeta()[getNumberOfLayers()-1]));
		delta[getNumberOfLayers()-1] = cost.delta(getActivations()[getNumberOfLayers()-1], y, getZeta()[getNumberOfLayers()-1]);
		for (int i = numberOfLayers-2; i >=1 ; i--) {
			//every activation array has a number of inputs as dimension. the other dimension matches the number neurons in each layer (need to add one for the bias)
			delta[i] = (delta[i+1].multiply( getWeights()[i])).hadamardProduct(sigmoidPrime(getZeta()[i]));//delta[i] = W[i]*delta(i+1).*sigmoidPrime(z[i])


		}
		//System.out.println("gradients");
		for (int i = 0; i< getNumberOfLayers()-1 ; i++) {
			gradients[0][i] = getActivations()[i].transpose().multiply(delta[i+1]);// wGrad = a(i)*delta(i+1)
			//gradients[1][i] = delta[i+1];add it below cause need to have 1 row

		}
		
		
		//multiply delta with unary array to make each row add to each other and get an 1Xdelta[i+1].colums() dimensional gradient
		for (int i = 0; i< getNumberOfLayers()-1 ; i++) {
			Matrix unary = Matrix.unit(1, delta[i+1].rows());
			gradients[1][i] = unary.multiply(delta[i+1]);
		}
		
		return gradients;
		
	}
	
	public int evaluate(){
		double start = System.nanoTime();
		double[] time = feedforward(test_inputs);
		double elapsedTime = System.nanoTime() - start;
		System.out.println(elapsedTime/1000000000+" seconds feedforward");
		double start2 = System.nanoTime(); 
		int corrects = 0;
		for (int j = 0; j < getActivations()[getNumberOfLayers()-1].rows(); j++) {
			
		
			Vector v = getActivations()[getNumberOfLayers()-1].getRow(j);
			int pos = maxPosition(v);
			if(test_outputs.get(j, pos)==1){
				corrects++;
			}
		
		}
		double elapsedTime2 = System.nanoTime() - start2;
		for (int i = 0; i < time.length; i++) {
			System.out.println(time[i]/1000000000+" seconds feed forward at layer :"+i);
		}
		
		System.out.println(elapsedTime2/1000000000+" seconds evaluation");
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

		Matrix sig = t.transform(new MatrixFunction() {
			
			@Override
			public double evaluate(int arg0, int arg1, double arg2) {
				return 1.0 / (1.0 + Math.exp(-arg2));
			}
		});
		
/*		for (int i = 0; i < t.rows(); i++) {
			for (int j = 0; j < t.columns(); j++) {
				sig.set(i, j, 1.0 / (1.0 + Math.exp(-t.get(i,j))));
			}
		}*/

		
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
		//System.out.println(m);
		System.out.println("Diamensions of matrix are "+m.rows()+"X"+m.columns() );
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

	public Matrix[] getBiases() {
		return biases;
	}

	public void setBiases(Matrix[] biases) {
		this.biases = biases;
	}

	public Matrix[] getZeta() {
		return zeta;
	}

	public void setZeta(Matrix[] zeta) {
		this.zeta = zeta;
	}
	
	public interface CostFunction{
		
		public double cost(Matrix a, Matrix y);
		public Matrix delta(Matrix a, Matrix y, Matrix z);
			
	}
	
	public class MeanSqueareErrorCostFunction implements CostFunction{

		@Override
		public double cost(Matrix a, Matrix y) {
			double cost=0.0;
			// least square cost
			Matrix  m = a.subtract(y); //a-y
			m = m.hadamardProduct(m);//(a-y)^2
			double sum=0.0;
			for (int i = 0; i < m.rows(); i++) {
				sum= sum +m.getRow(i).sum();
			}
			cost = sum/2;//1/2n * sum((a-y)^2)
			
			return cost;
		}

		//delta for the last layer
		@Override
		public Matrix delta(Matrix a, Matrix y, Matrix z) {
			return a.subtract(y).hadamardProduct(sigmoidPrime(z));		
		}
		
	}
	
	public class CrossEntropyCostFunction implements CostFunction{

		@Override
		public double cost(Matrix a, Matrix y) {
			Matrix cost = (negative(y.hadamardProduct(log(a)))).subtract((Matrix.unit(y.rows(), y.columns()).subtract(y)).hadamardProduct(log(Matrix.unit(a.rows(), a.columns()).subtract(a))));
			return cost.sum();
		}
		//delta for the last layer
		@Override
		public Matrix delta(Matrix a, Matrix y, Matrix z) {
			return a.subtract(y);
		}
		private Matrix negative(Matrix m){
			return m.transform(new MatrixFunction() {
				
				@Override
				public double evaluate(int arg0, int arg1, double arg2) {
					return -arg2;
				}
			});
		}
		private Matrix log(Matrix m){
			return m.transform(new MatrixFunction() {
				
				@Override
				public double evaluate(int arg0, int arg1, double arg2) {
					return Math.log10(arg2);
				}
			});
		}
	}
	

}


