import java.util.*; 

public class NNetwork{
	public static final int INPUT_LAYER = 0; 
	public static final int HIDDEN_LAYER = 1;
	public static final int OUTPUT_LAYER = 2;
	public static final int PIXEL_NUM = 28;
	
	private double learningRate;
	private Vector<NNLayer> layers;
	private int numLayers;
	private int numInputNodes;
	private int numOutputNodes;
	private int numHiddenNodes;
	private NNLayer inputLayer;
	private NNLayer outputLayer; 
	private int[][] inputArray;
	
	public double[] error;
	private double[] outputs; //actual values for each node in output layer
	private double[] expectedOutputs; //target values for each node in output layer
	
    //Updates the error
	public void updateError(){
	if(this.outputs.length == this.expectedOutputs.length){
			for (int i = 0; i < this.outputs.length;i++){
				this.error[i] = this.expectedOutputs[i] - this.outputs[i];
			}
		}
	}
    //Adds the layer to the NNetwork
	public void addLayer(NNLayer layer){
		// First layer added is set as input and output layer
		if(this.layers.isEmpty()){
			this.layers.add(layer); 
			this.inputLayer = layer; 
			this.outputLayer = this.layers.elementAt(0);
	
		}
		// Subsequent layers added are set as output layer and connected to previous layer
		else{
			this.layers.add(layer);
			this.outputLayer = this.layers.elementAt(this.layers.size()-1);
			layer.connectLayer(this.layers.elementAt(this.layers.size()-2));
		}		
	}
    //propagates the new input forward through the network and updates the outputs array
	public void feedForward(){
		for (int i = 0; i < this.numLayers; i++){
			this.layers.elementAt(i).activate();
		}
		for(int i = 0; i < this.numOutputNodes; i++){
			outputs[i] = this.outputLayer.getNodeAt(i).getOutput();
		}	
	}
	
	//updating the weights after feedForward
	public void backPropagate(){
		//all derivatives computed during feed forward
		//update weights for hidden-to-output nodes:
		for(int i = 0; i < this.numOutputNodes; i++){
		    for(int j = 0; j < this.numHiddenNodes; j++){
			double w = this.outputLayer.getNodeAt(i).getWeightAt(j);
			w += this.learningRate * this.outputLayer.getNodeAt(i).getInputAt(j) * error[i] * this.outputLayer.getNodeAt(i).getDerivative();
			this.outputLayer.getNodeAt(i).setWeightAt(j,w);
		    }
		}
		
		//update weights for input-to-hidden nodes:
		for(int l = this.layers.size()-2; l > 0; l--) {
		    for(int j = 1; j < this.numHiddenNodes; j++){
                double sum = 0;
                for( int i = 0; i < this.numOutputNodes; i++){
                    sum += (this.outputLayer.getNodeAt(i).getWeightAt(j) * error[i] * this.outputLayer.getNodeAt(i).getDerivative());
                }
                for(int k = 0; k < this.numInputNodes; k++){
                    double w = this.layers.elementAt(l).getNodeAt(j).getWeightAt(k);
                    w += this.learningRate * this.layers.elementAt(l).getNodeAt(j).getInputAt(k) * sum * this.layers.elementAt(l).getNodeAt(j).getDerivative();
                    this.layers.elementAt(l).getNodeAt(j).setWeightAt(k,w); 
                }
		    }
		}
	}
    //sets all the weights (used when testing from previously calculated weights)
    public void setAllWeights(double[] weights) {
        int count = 0;
        for (int i = 1; i < this.numHiddenNodes; i++) {
            for (int j = 0; j < this.numInputNodes; j++){
                this.getLayerAt(1).getNodeAt(i).setWeightAt(j,weights[count]);
                count++;
            }
        }
        for (int i = 0; i < this.numOutputNodes; i++) {
            for (int j = 0; j < this.numHiddenNodes; j++){
                this.getLayerAt(2).getNodeAt(i).setWeightAt(j,weights[count]);
                count++;
            }
        }
    }
    //returns the outputs
    public double[] getOutputs(){
        return this.outputs;
    }
    //sets the outputs
	public void setTargets(double[] out){
		for(int i = 0; i < expectedOutputs.length; i++){
			expectedOutputs[i] = out[i];
		}
	}
    //sets the inputs
	public void setInputs(int[][] in){
		this.setInputArray(in);
		this.insertInputs(this.formatInput());
	}
	//sets the input array
	public void setInputArray(int [][] in){
		this.inputArray = in;
	}
	//converts 2D array of integers between 0-255 to a 1D array of doubles between 0 and 1
	public double[] formatInput(){
		double [] finput = new double[PIXEL_NUM*PIXEL_NUM];
		int count = 0;
		for(int i = 0; i < PIXEL_NUM; i++){
			for (int j = 0 ; j <PIXEL_NUM; j++){ 
				finput[count] = (double)this.inputArray[i][j] / 255.0;
				count++;
			}
		}
		return finput;
	}
	//setInputs helper
	public void insertInputs(double[] in){
		this.inputLayer.setInputs(in);
	}

	//returns node i in the layer
	public NNLayer getLayerAt(int i) {
		return this.layers.elementAt(i);	
	}
    //sets a random learning rate
    public void randLearningRate() {
        Random r = new Random();
        double randomValue = 0.015 + (0.15 - 0.015) * r.nextDouble();
        this.learningRate = randomValue;
    }
    //updates the learning rate for a dynamic learning rate
    public void dynamicLearningRate(int r) {
        this.learningRate -= this.learningRate/(r*.33);
    }

	//debugging print statement
	public void printNetwork(){
//		System.out.print("input layer");
//		for (int i = 0; i < this.inputLayer.getNumNodes(); i++){
//			this.inputLayer.getNodeAt(i).printNode(); 
//		}
//		System.out.println();
//		System.out.print("hidden layer: ");
//		for(int i = 0; i < (this.layers.elementAt(1).getNumNodes())%100; i++){
//			this.layers.elementAt(1).getNodeAt(i).printNode();
//		}
//		System.out.println(); 
//		System.out.print("output layer: "); 
//		for(int i = 0; i < this.layers.elementAt(2).getNumNodes(); i++){
//			this.layers.elementAt(2).getNodeAt(i).printNode();
//		}
		System.out.println();
		for(int i = 0; i < this.outputs.length; i++){
			System.out.println("output["+i+"] : " + this.outputs[i]);
			System.out.println("expected output["+i+"] : " +this.expectedOutputs[i]);
		}
		System.out.println(); 
		System.out.println(); 
		System.out.println(); 
		System.out.println(); 
		
	}
	// Constructor: number of layers, input nodes, hidden nodes(per layer), output nodes
	public NNetwork(int l, int inN, int hidN, int outN, double lr){
		this.layers = new Vector<NNLayer>();
		this.numLayers = l;
		this.numInputNodes = inN;
		this.numHiddenNodes = hidN;
		this.numOutputNodes = outN;
		this.error = new double[outN];
		this.outputs = new double[outN];
		this.expectedOutputs = new double[outN];
		this.learningRate = lr;
        
		for(int i = 0; i < l; i++){
			NNLayer layer = new NNLayer(i);
			this.addLayer(layer);
		}
		for(int i = 0; i <inN; i++){
		    this.inputLayer.addNode();
		}
		
		for(int i = 0; i < l-2; i++){ //for each hidden layer
			for(int j = 0; j <hidN; j++){
				this.layers.elementAt(i+1).addNode();
			}
		}
        
		for(int i = 0; i < outN; i++){
			this.outputLayer.addNode();
		}
	}
}
