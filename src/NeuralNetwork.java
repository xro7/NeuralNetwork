import org.la4j.Matrix;

public class NeuralNetwork {
	
	private int numberOfLayers;
	private int[] layers;
	private Matrix[] weights;
	private Matrix[] activations;
	
	public NeuralNetwork(int[] layers,Matrix x){
		this.numberOfLayers = layers.length;
		this.setLayers(layers);
		this.weights = new Matrix[numberOfLayers-1];
		for (int i = 0; i < numberOfLayers -1 ; i++) {
			this.weights[i] = Matrix.zero(layers[i+1], layers[i] + 1 /*(this is the bias term that added to the weights)*/);
			randomInit(this.weights[i]);
		}
		activations = new Matrix[layers.length];// first layer is the inputs
		activations[0] = x.copy();
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
	
	private static void randomInit(Matrix matrix){
		java.util.Random r = new java.util.Random();
		for (int i = 0; i < matrix.rows(); i++) {
			for (int j = 0; j < matrix.columns(); j++) {
				matrix.set(i, j, r.nextGaussian());
			}
		}
	}
	
	private static void sigmoid(Matrix t){
		for (int i = 0; i < t.rows(); i++) {
			for (int j = 0; j < t.columns(); j++) {
				t.set(i, j, 1.0 / (1.0 + Math.exp(-t.get(i,j))));
			}
		}
		
	}
	
	public void feedforward(Matrix[] activations){
		
		for (int i = 1; i < numberOfLayers ; i++) {
			activations[i] = Matrix.zero(activations[0].rows(), layers[i]);//need to add a column of one every time i create an activation
		}
		
		
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
	
	

}
