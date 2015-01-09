import java.util.Vector;

public class NNLayer{
	public static final int INPUT_LAYER = 0; 
	public static final int HIDDEN_LAYER = 1;
	public static final int OUTPUT_LAYER = 2;
	
	private Vector<NNNode> nodes;
	private NNLayer previous; 
	private int layerType; 
	
    //Constructor, takes in layer type
	public NNLayer(int lType){
		NNNode biasNode = new NNNode(lType); 
		biasNode.setInput(1.0);
		this.layerType = lType; 
		if(lType == HIDDEN_LAYER){
			this.nodes = new Vector<NNNode>();
			this.nodes.add(biasNode); 
		}
		else{
			this.nodes = new Vector<NNNode>();
		}
	}
    //Connects current layer to the previous layer
	public void connectLayer(NNLayer p){
		this.previous = p; 
	}
    //adds new node to vector of nodes in layer and connects this node to all nodes in previous layer
	public void addNode(){
		NNNode n = new NNNode(this.layerType);
		this.nodes.add(n);
		if(this.layerType != INPUT_LAYER){
			for (int j = this.layerType-1; j < this.previous.nodes.size(); j ++){
				NNConnection c = new NNConnection(this.previous.nodes.elementAt(j));
				n.connect(c);
			}
		}
	}
    //sets the input for the input layer only
	public void setInputs(double[] in){
		for(int i = 0; i < this.nodes.size(); i++){
			this.nodes.elementAt(i).setInput(in[i]); 
		}
	}
	//for each node: updates stored values for weighted input, output, and derivative
	public void activate(){
		for (int i = 0; i < this.nodes.size(); i++){
			if(this.layerType == OUTPUT_LAYER){this.getNodeAt(i).computeWeightedInput();} 
			if(this.layerType == HIDDEN_LAYER && i > 0){this.getNodeAt(i).computeWeightedInput();}
			this.nodes.elementAt(i).computeOutput();
			this.nodes.elementAt(i).computeDerivative();
		}
	}
	//returns the number of nodes in the layer
	public int getNumNodes(){
		return this.nodes.size();
	}
    //returns the nodes in the layer
	public Vector<NNNode> getNodes(){
		return this.nodes; 
	}
    //returns the node at position i
	public NNNode getNodeAt(int i){
		return this.nodes.elementAt(i);
	}
    //debuging print statement
	public void printLayer(){
		System.out.print("layer type: " + this.layerType);
		System.out.println(" with " +this.nodes.size() + "nodes"); 
		System.out.println("All nodes: "); 
		for(int n = 0; n < this.nodes.size(); n++){
			this.nodes.elementAt(n).printNode(); 
		}
	}
}
