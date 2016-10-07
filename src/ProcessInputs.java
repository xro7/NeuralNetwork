import java.io.IOException;
import java.nio.file.Paths;
import java.util.InputMismatchException;
import java.util.Locale;
import java.util.Scanner;

import org.la4j.Matrix;



public class ProcessInputs {
	
	private String path;
	private int instances;
	private int features;
	private Matrix inputs;
	
	public ProcessInputs(int instances, int features, String path){
		this.path = path;
		this.features = features;
		this.instances = instances;
		inputs = Matrix.constant(instances, features, 0.0);
		createMatrix(inputs);
	}

	public String getPath() {
		return path;
	}

	public void setPath(String path) {
		this.path = path;
	}

	public Matrix getInputs() {
		return inputs;
	}

	public void setInputs(Matrix inputs) {
		this.inputs = inputs;
	}

	public int getFeatures() {
		return features;
	}

	public void setFeatures(int features) {
		this.features = features;
	}

	public int getInstances() {
		return instances;
	}

	public void setInstances(int instances) {
		this.instances = instances;
	}
	
	private void createMatrix(Matrix inputs){
		 int i = 0;
		 try (Scanner scanner =  new Scanner(Paths.get(getPath()), "UTF-8").useLocale(Locale.ENGLISH);){
		    	while (scanner.hasNextLine()){	  		    		
		    	    try (Scanner s = new Scanner(scanner.nextLine())){;
		    	    	s.useLocale(Locale.ENGLISH);
		    	    	if (i == getInstances()){
		    	    		break;
		    	    	}
		    	    	s.useDelimiter("\\s");
		    	    	int j = 0;
		    		    while (s.hasNext()){
		    		    	try{
		    		    	inputs.set(i, j, s.nextDouble());
		    		    	
			    		    } catch (InputMismatchException e) {
			    		        System.out.print(e.getMessage()); //try to find out specific reason.
			    		    }
		    		    	j++;
			    	    	if (j == getFeatures()){
			    	    		break;
			    	    	}
		    		    }    
		    		    i++;
		    	    }	    	  
		    	}		 		    	    
		    } catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
	}
	

}
