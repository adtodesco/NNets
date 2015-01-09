import java.util.Vector;

public class NNNode{
	public static final int INPUT_LAYER = 0; 
	public static final int HIDDEN_LAYER = 1;
	public static final int OUTPUT_LAYER = 2;
	
	
	private Vector<NNConnection> inputs;//vector of connections TO this node
	private double weightedInput; //sum of all weighted inputs to the node
	private double output;//output calculated by activation function
	private double derivative; //derivative as calculated by backpropagation algorithm 
	
	private double sigmoidActivate(double wIn){//activation function
		return 1.0/ (1 + Math.exp(-1.0 * wIn));
	}
	
	private double sigmoidDerive(){//equation for calculation the derivative
		return this.output * (1 - this.output); 
	} 
	
	public void computeOutput(){//applies activation function to weighted input to generate output value
		this.output = sigmoidActivate(this.weightedInput);
	}
	public void setInput(double in){//for input nodes and bias nodes
		this.weightedInput = in;
	}
	public void computeWeightedInput(){//computes a weighted sum of all input connections to the node (for all other nodes)
		double wIn = 0; 
		for (int i = 1; i < this.inputs.size(); i ++){
			wIn += this.inputs.elementAt(i).getWeightedInput(); // inputs is a vector of connections, so getWeightedInput() doesn't work!
		}
		this.weightedInput = wIn;
	}
	public void computeDerivative(){//computes the derivative of the node
		this.derivative = sigmoidDerive();
	}
	public double getOutput(){//used in NNConnection
		return this.output; 
	}
	public double getDerivative(){//returns the derivate of the output
		return this.derivative; 
	}
	public void connect(NNConnection c){//connects this node to an input to this node (connects BACKWARDS)
		inputs.add(c);
	}
	public void setWeightAt(int i, double w){//sets the weight for connection from node i to this node to be w
		this.inputs.elementAt(i).setWeight(w);
	}
	public double getWeightedInput(){//returns the weight input
		return this.weightedInput;
	}
	public double getWeightAt(int i){//returns the weight of connection
		return this.inputs.elementAt(i).getWeight();
	}
	public double getInputAt(int i){//returns input value of the node
		return this.inputs.elementAt(i).getInput(); 
	}
	
	public NNNode(int l){//creates a new node in of given layer type
		if(l != INPUT_LAYER){
			this.inputs = new Vector<NNConnection>(); 
		}
		else{
			this.inputs = new Vector<NNConnection>(1); 
		}
	}
    //Debuging print statement
	public void printNode(){
		System.out.println("input :" + this.weightedInput);
		
		if(!this.inputs.isEmpty()){
			for(int i = 0; i < this.inputs.size(); i++){
				this.inputs.elementAt(i).printConnection(); 
			}
		}
		else{System.out.println("no connections");}
	}

}
