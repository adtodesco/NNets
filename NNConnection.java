public class NNConnection{
	private NNNode inNode;
	private double weight;
	
	public NNConnection(NNNode n){//constructor with random SMALL weights initialized between 0.0 and 1.0/3
		this.inNode = n;
		this.weight = (Math.random()/3.0);
		if(Math.random()<0.5){
			this.weight = this.weight*-1.0;
		}
	}
	public double getWeight(){//returns the weight of the connection
		return this.weight;
	}
	public void setWeight(double w){//sets the weight of the connection
		this.weight = w;
	}
	public double getInput(){//returns the output of the in-coming node
		return this.inNode.getOutput();
	}
	public double getWeightedInput(){//calculates the weighted input of the connection
        return (this.inNode.getOutput() * this.weight);
    }
    public void printConnection(){//debugging print statement
        System.out.println("weight of connection: " + this.weight);
        System.out.println("input from connection: " + this.getInput());
    }
}
