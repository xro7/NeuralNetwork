import org.la4j.Matrix;
import org.la4j.Vector;
import org.la4j.matrix.DenseMatrix;
import org.la4j.matrix.functor.MatrixFunction;

public class NeuralNetwork {
	
	private int numberOfLayers;
	private int[] layers;
	private DenseMatrix[] weights;
	private DenseMatrix[] biases;
	private DenseMatrix[] activations;
	private DenseMatrix[] zeta;
	private DenseMatrix training_inputs;
	private DenseMatrix training_outputs;
	private DenseMatrix test_inputs;
	private DenseMatrix test_outputs;
	
	public NeuralNetwork(int[] layers,DenseMatrix trainSet,DenseMatrix validationSet,DenseMatrix testSet){
		this.numberOfLayers = layers.length;
		this.setLayers(layers);
		this.activations = new DenseMatrix[layers.length];
		this.zeta = new DenseMatrix[layers.length];
		this.weights = new DenseMatrix[numberOfLayers-1];
		this.biases = new DenseMatrix[numberOfLayers-1];
		this.setTraining_outputs(rawValuesToVector((DenseMatrix) trainSet.getColumn(0).toColumnMatrix(), layers[numberOfLayers-1]));
		this.setTraining_inputs((DenseMatrix) trainSet.removeFirstColumn());
		this.test_outputs = rawValuesToVector((DenseMatrix) testSet.getColumn(0).toColumnMatrix(), layers[numberOfLayers-1]);
		this.test_inputs = (DenseMatrix) testSet.removeFirstColumn();
		
		for (int i = 0; i < numberOfLayers -1 ; i++) {
			this.weights[i] = DenseMatrix.zero(layers[i+1], layers[i]);
			this.biases[i] = DenseMatrix.zero(1, layers[i+1]);
			randomInitWeights(this.weights[i]);
			randomInitBiases(this.biases[i]);
		}

		//sgd(30,10,0.001,new CrossEntropyCostFunction(10.0),new Sigmoid(),true);
		
	
	}
	
	public void feedforward(DenseMatrix x,ActivationFunction activationFunction){
		
		activations[0] = x;		
		DenseMatrix[] bias = new DenseMatrix[layers.length-1];
		for (int i = 1; i < numberOfLayers ; i++) {
			
			DenseMatrix unary = DenseMatrix.unit(getActivations()[i-1].rows(), 1);
			bias[i-1] = (DenseMatrix) unary.multiply(getBiases()[i-1]);//need to make bias[i] from 1 X layers[i] to getActivations()[i-1].rows() X layers[i] so as to add it to z
			zeta[i] = (DenseMatrix) (getActivations()[i-1].multiply(getWeights()[i-1].transpose())).add(bias[i-1]) ;//z =W*a b	
			getActivations()[i] = activationFunction.activation(getZeta()[i]);//sigmoid(z)
			
		}			
	}		
	
	public void sgd(int epochs,int mini_batch_size,double eta,CostFunction cost,ActivationFunction activationFunction,boolean printTrainingCost){
		
		if (printTrainingCost){
			feedforward(getTraining_inputs(),activationFunction);
			System.out.println("Initial Cost of "+ cost.toString()+" is:" + cost.cost(getActivations()[getNumberOfLayers()-1], getTraining_outputs()));
		}
		
		System.out.println(evaluate(activationFunction)+" /"+test_inputs.rows());//evaluate test set with random weights
		int batches = getTraining_inputs().rows() / mini_batch_size;//number of batches		 
		DenseMatrix[][] accumulators = new DenseMatrix[2][layers.length-1];

		for (int e = 0; e < epochs; e++) {
			
			System.out.println("epoch: "+(e+1));
			double start = System.nanoTime();   
						
			for (int b = 0; b < batches; b++) {
				DenseMatrix x = (DenseMatrix) getTraining_inputs().slice(b*mini_batch_size, 0,(b+1)*mini_batch_size, getTraining_inputs().columns());
				DenseMatrix y = (DenseMatrix) getTraining_outputs().slice(b*mini_batch_size, 0,(b+1)*mini_batch_size, getTraining_outputs().columns());

				accumulators = backpropagation(x, y,cost,activationFunction);
	
				//divide gradients by 1/mini_batch_size
				for (int j = 0; j < layers.length-1; j++) {
					accumulators[0][j] = (DenseMatrix) accumulators[0][j].multiply((double)1/mini_batch_size);
					accumulators[1][j] = (DenseMatrix) accumulators[1][j].multiply((double)1/mini_batch_size);
				}
				
				//update weights and bias subtracting the gradients multiplied by learning rate
				for (int i = 0; i <layers.length-1; i++) {
					weights[i] = (DenseMatrix) (weights[i].multiply((1-eta*(cost.getLambda()/getTraining_inputs().rows())))).subtract(accumulators[0][i].multiply(eta).transpose());
					biases[i] = (DenseMatrix) biases[i].subtract(accumulators[1][i].multiply(eta));
				}			
			}
			
			if (printTrainingCost){
				feedforward(getTraining_inputs(),activationFunction);
				System.out.println("Cost of "+ cost.toString()+" is:" + cost.cost(getActivations()[getNumberOfLayers()-1], getTraining_outputs()));
			}
			
			double elapsedTime = System.nanoTime() - start;
			System.out.println(elapsedTime/1000000000+" seconds for this training epoch");
			System.out.println(evaluate(activationFunction)+" /"+test_inputs.rows());

		}
		//System.out.println(evaluate()+" /"+test_inputs.rows());//evaluate test set with random weights
		
	}
	
	public DenseMatrix[][] backpropagation(DenseMatrix x, DenseMatrix y,CostFunction cost,ActivationFunction activationFunction){
		//feed forward 	
		feedforward(x,activationFunction);
		
		DenseMatrix[][] gradients = new DenseMatrix[2][layers.length-1];// for weights and bias
		DenseMatrix[] delta = new DenseMatrix[layers.length];
		
		//find deltas
		delta[getNumberOfLayers()-1] = cost.delta(getActivations()[getNumberOfLayers()-1], y, getZeta()[getNumberOfLayers()-1]);
		for (int i = numberOfLayers-2; i >=1 ; i--) {
			//every activation array has a number of inputs as dimension. the other dimension matches the number neurons in each layer (need to add one for the bias)
			delta[i] = (DenseMatrix) (delta[i+1].multiply( getWeights()[i])).hadamardProduct(activationFunction.activationPrime(getZeta()[i]));//delta[i] = W[i]*delta(i+1).*sigmoidPrime(z[i])


		}
		//System.out.println("gradients");
		for (int i = 0; i< getNumberOfLayers()-1 ; i++) {
			gradients[0][i] = (DenseMatrix) getActivations()[i].transpose().multiply(delta[i+1]);// wGrad = a(i)*delta(i+1)
			//gradients[1][i] = delta[i+1];add it below cause need to have 1 row

		}
		
		
		//multiply delta with unary array to make each row add to each other and get an 1Xdelta[i+1].columns() dimensional gradient
		for (int i = 0; i< getNumberOfLayers()-1 ; i++) {
			DenseMatrix unary = DenseMatrix.unit(1, delta[i+1].rows());
			gradients[1][i] = (DenseMatrix) unary.multiply(delta[i+1]);
		}
		
		return gradients;
		
	}
	
	public int evaluate(ActivationFunction activationFunction){
		feedforward(test_inputs,activationFunction);
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
	
/*	public DenseMatrix softmax(DenseMatrix outputActivations){
		
		double sum=0.0;
		for (int i = 0; i < outputActivations.rows(); i++) {
			sum = sum + Math.exp(outputActivations.get(i, 0));
		}
		
		DenseMatrix softmax = DenseMatrix.zero(outputActivations.rows(), 1);
		for (int i = 0; i < softmax.rows(); i++) {
			softmax.set(i, 0, Math.exp(outputActivations.get(i, 0))/sum);
		}
		return softmax;

	}*/
	
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
	
	private static void randomInitWeights(DenseMatrix matrix){
		java.util.Random r = new java.util.Random();
		for (int i = 0; i < matrix.rows(); i++) {
			for (int j = 0; j < matrix.columns(); j++) {
				matrix.set(i, j, r.nextGaussian()/Math.sqrt(matrix.columns()));
			}
		}
	}
	
	private static void randomInitBiases(DenseMatrix matrix){
		java.util.Random r = new java.util.Random();
		for (int i = 0; i < matrix.rows(); i++) {
			for (int j = 0; j < matrix.columns(); j++) {
				matrix.set(i, j, r.nextGaussian());
			}
		}
	}
	
	private DenseMatrix sigmoid(DenseMatrix t){

		DenseMatrix sig = (DenseMatrix) t.transform(new MatrixFunction() {
			
			@Override
			public double evaluate(int arg0, int arg1, double arg2) {
				return 1.0 / (1.0 + Math.exp(-arg2));
			}
		});
				
		return(sig);
	
	}
	
	private DenseMatrix sigmoidPrime(DenseMatrix z){
		
		DenseMatrix prime = (DenseMatrix) sigmoid(z).hadamardProduct(DenseMatrix.constant(z.rows(), z.columns(), 1.0).subtract(sigmoid(z)));
		return prime;
	}
	
	
	private DenseMatrix rawValuesToVector(DenseMatrix m, int sizeVector){
		DenseMatrix y = DenseMatrix.zero(m.rows(), sizeVector) ;
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

	public DenseMatrix[] getWeights() {
		return weights;
	}

	public void setWeights(DenseMatrix[] weights) {
		this.weights = weights;
	}
	
	public DenseMatrix[] getActivations() {
		return activations;
	}

	public void setActivations(DenseMatrix[] activations) {
		this.activations = activations;
	}
	
	public static void printDimensions(DenseMatrix m){
		//System.out.println(m);
		System.out.println("Diamensions of matrix are "+m.rows()+"X"+m.columns() );
	}
	
	public DenseMatrix getTraining_inputs() {
		return training_inputs;
	}

	public void setTraining_inputs(DenseMatrix training_inputs) {
		this.training_inputs = training_inputs;
	}

	public DenseMatrix getTraining_outputs() {
		return training_outputs;
	}

	public void setTraining_outputs(DenseMatrix training_outputs) {
		this.training_outputs = training_outputs;
	}

	public DenseMatrix[] getBiases() {
		return biases;
	}

	public void setBiases(DenseMatrix[] biases) {
		this.biases = biases;
	}

	public DenseMatrix[] getZeta() {
		return zeta;
	}

	public void setZeta(DenseMatrix[] zeta) {
		this.zeta = zeta;
	}
	
	public interface CostFunction{
		
		
		public double cost(DenseMatrix a, DenseMatrix y);
		public DenseMatrix delta(DenseMatrix a, DenseMatrix y, DenseMatrix z);
		public double getLambda();
	}
	
	public class MeanSqueareErrorCostFunction implements CostFunction{
		
		private double lambda;
		
		public MeanSqueareErrorCostFunction(double lambda){
			this.lambda = lambda;
		}

		@Override
		public double cost(DenseMatrix a, DenseMatrix y) {
			double cost=0.0;
			// least square cost
			DenseMatrix  m = (DenseMatrix) a.subtract(y); //a-y
			m = (DenseMatrix) m.hadamardProduct(m);//(a-y)^2
			double sum=0.0;
			for (int i = 0; i < m.rows(); i++) {
				sum= sum +m.getRow(i).sum();
			}
			cost = sum/2;//1/2 * sum((a-y)^2)
			double regsum = 0.0;
			for (int i = 0; i < getNumberOfLayers()-1; i++) {
				Matrix w = getWeights()[i].hadamardProduct(getWeights()[i]);
				regsum  += w.sum ();
			}
			double regularazationPenalty = (lambda/(2*getTraining_inputs().rows())) *regsum ;
			
			return cost+regularazationPenalty;
		}

		//delta for the last layer
		@Override
		public DenseMatrix delta(DenseMatrix a, DenseMatrix y, DenseMatrix z) {
			return (DenseMatrix) a.subtract(y).hadamardProduct(sigmoidPrime(z));		
		}
		
		@Override
		public String toString() {
			return "Mean Square Error Cost Function";
		}
		@Override
		public double getLambda(){
			return lambda;
		}
		
	}
	
	public class CrossEntropyCostFunction implements CostFunction{
		
		private double lambda;
		
		public CrossEntropyCostFunction(double lambda){
			this.lambda = lambda;
		}

		@Override
		public double cost(DenseMatrix a, DenseMatrix y) {
			DenseMatrix cost = (DenseMatrix) ((negative((DenseMatrix) y.hadamardProduct(log(a)))).subtract((DenseMatrix.unit(y.rows(), y.columns()).subtract(y)).hadamardProduct(log((DenseMatrix) DenseMatrix.unit(a.rows(), a.columns()).subtract(a)))));
			double sum = 0.0;
			for (int i = 0; i < getNumberOfLayers()-1; i++) {
				Matrix w = getWeights()[i].hadamardProduct(getWeights()[i]);
				sum += w.sum();
			}
			double regularazationPenalty = (lambda/(2*getTraining_inputs().rows())) *sum;
			return cost.sum()+regularazationPenalty ;
		}
		//delta for the last layer
		@Override
		public DenseMatrix delta(DenseMatrix a, DenseMatrix y, DenseMatrix z) {
			return (DenseMatrix) a.subtract(y);
		}
		private DenseMatrix negative(DenseMatrix m){
			return (DenseMatrix) m.transform(new MatrixFunction() {
				
				@Override
				public double evaluate(int arg0, int arg1, double arg2) {
					return -arg2;
				}
			});
		}
		private DenseMatrix log(DenseMatrix m){
			return (DenseMatrix) m.transform(new MatrixFunction() {
				
				@Override
				public double evaluate(int arg0, int arg1, double arg2) {
					return Math.log10(arg2);
				}
			});
		}
		
		@Override
		public String toString() {
			return "Cross Entropy Cost Function";
		}
		@Override
		public double getLambda(){
			return lambda;
		}
	}
	
	public interface ActivationFunction{
		public DenseMatrix activation(DenseMatrix z);
		public DenseMatrix activationPrime(DenseMatrix z);
	}
	
	public class Sigmoid implements ActivationFunction{

		@Override
		public DenseMatrix activation(DenseMatrix z) {
			DenseMatrix sig = (DenseMatrix) z.transform(new MatrixFunction() {
				
				@Override
				public double evaluate(int arg0, int arg1, double arg2) {
					return 1.0 / (1.0 + Math.exp(-arg2));
				}
			});
					
			return(sig);
		}

		@Override
		public DenseMatrix activationPrime(DenseMatrix z) {
			DenseMatrix prime = (DenseMatrix) activation(z).hadamardProduct(DenseMatrix.constant(z.rows(), z.columns(), 1.0).subtract(activation(z)));
			return prime;
		}
		
	}
	
	public class Tanh implements ActivationFunction{

		@Override
		public DenseMatrix activation(DenseMatrix z) {
			DenseMatrix sig = (DenseMatrix) z.transform(new MatrixFunction() {
				
				@Override
				public double evaluate(int arg0, int arg1, double arg2) {
					return  (2.0 / (1.0 + Math.exp(-2.0*arg2))) -1.0;
				}
			});
					
			return(sig);
		}

		@Override
		public DenseMatrix activationPrime(DenseMatrix z) {
			DenseMatrix prime =  (DenseMatrix) DenseMatrix.constant(z.rows(), z.columns(), 1.0).subtract(activation(z).hadamardProduct(activation(z)));
			return prime;
		}
		
	}
	
	public class ReLU implements ActivationFunction{

		@Override
		public DenseMatrix activation(DenseMatrix z) {
			DenseMatrix sig = (DenseMatrix) z.transform(new MatrixFunction() {
				
				@Override
				public double evaluate(int arg0, int arg1, double arg2) {
					return  Math.max(0, arg2);
				}
			});
					
			return(sig);
		}

		@Override
		public DenseMatrix activationPrime(DenseMatrix z) {
			DenseMatrix prime =  (DenseMatrix) z.transform(new MatrixFunction() {
				
				@Override
				public double evaluate(int arg0, int arg1, double arg2) {
					if (arg2>0){
						return  1.0;
					}
					else{
						return  0.0;
					}
				}
			});
			return prime;
		}
		
	}
	

}


