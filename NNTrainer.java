import java.util.*;
import java.io.*;

public class NNTrainer{
    public static final int INPUT_LAYER = 0;
	public static final int HIDDEN_LAYER = 1;
	public static final int OUTPUT_LAYER = 2;
	public static final int PIXEL_NUM = 28;
    
    public static final int TRAINING_NUM = 30000;
    private int NUM_IMAGES;
    public NNetwork net;
    private int[][][] inputs;
    private int[] targets;
    private MNISTReader reader;
    
    public static final int NUM_LAYERS = 3;
    public static final int NUM_IN_NODES = 784;
    public static final int NUM_HID_NODES = 50;
    public static final int NUM_OUT_NODES = 10;
    public static int NUM_EPOCHS = 10;
    public static int NUM_ITERATIONS = 10000;
    
    public NNTrainer(double lr, int numImages){
        this.NUM_IMAGES = numImages;
        
        this.inputs = new int[TRAINING_NUM][PIXEL_NUM][PIXEL_NUM];
        this.targets = new int[TRAINING_NUM];
        reader = new MNISTReader(inputs, targets, 0);
        net = new NNetwork(NUM_LAYERS,NUM_IN_NODES,NUM_HID_NODES,NUM_OUT_NODES,lr);
        try{
            reader.Read();
        }catch(IOException e){}
        
    }
    
    // training method
    // to train on the groups of different images, vary parameter based on epoch
    public void train(int i){
        // check for valid iteration size
//        if (i > NUM_IMAGES - NUM_ITERATIONS) {
//            System.out.println("Warning: Invalid number of iterations.");
//            return;
//        }
        // main training loop
        // for each iteration, set inputs, set target, feed forward, back propogate
        for(int iteration = i; iteration < i + NUM_ITERATIONS; iteration++){
            // get targets and inputs from reader
            int tar = this.targets[iteration];
            int[][] in = this.inputs[iteration];
            double[] tarArray;
            // for decimal style output
            if(NUM_OUT_NODES==10){
                // generate target array for this training epoch
                tarArray = new double[10];
                for (int t = 0; t < 10; t++){
                    if (t == tar){
                        tarArray[t] = 1.0;
                    }
                    else{
                        tarArray[t] = 0.0;
                    }
                }
            }
            // for binary style output
            else {
                tarArray = new double[4];
                //generate target array for this training epoch
                for (int t = 0; t < 4; t++){
                    tarArray[t] = 0.0;
                }
                // set targets appropriate for binary outputs
                switch(tar) {
                    case 1:
                        tarArray[0] = 1.0;
                        break;
                    case 2:
                        tarArray[1] = 1.0;
                        break;
                    case 3:
                        tarArray[0] = 1.0;
                        tarArray[1] = 1.0;
                        break;
                    case 4:
                        tarArray[2] = 1.0;
                        break;
                    case 5:
                        tarArray[0] = 1.0;
                        tarArray[2] = 1.0;
                        break;
                    case 6:
                        tarArray[1] = 1.0;
                        tarArray[2] = 1.0;
                        break;
                    case 7:
                        tarArray[0] = 1.0;
                        tarArray[1] = 1.0;
                        tarArray[2] = 1.0;
                        break;
                    case 8:
                        tarArray[3] = 1.0;
                        break;
                    case 9:
                        tarArray[0] = 1.0;
                        tarArray[3] = 1.0;
                        break;
                    default:
                        break;
                        
                }
            }
            // set the expected outputs for this training epoch
            this.net.setTargets(tarArray);
            // set the inputs for this training epoch
            this.net.setInputs(in);
            this.net.feedForward();
            this.net.updateError(); 
            this.net.backPropagate();
            
            // *** UNCOMMENT EITHER METHOD TO USE *** //
            // this.net.randLearningRate();
            // this.net.dynamicLearningRate(NUM_EPOCHS*ITERATIONS);
        }
    }

    // main method
    public static void main (String[] args){
        // input parameters for learning rate, iterations, and epochs
        double lr = Double.parseDouble(args[0]);
        int NUM_ITERATIONS = Integer.parseInt(args[1]);
        int NUM_EPOCHS = Integer.parseInt(args[2]);
        // initialize trainer
        NNTrainer T = new NNTrainer(lr, NUM_ITERATIONS);
        for( int epoch = 0; epoch < NUM_EPOCHS; epoch++){
        	T.train(0);
        	System.out.println("Epoch "+epoch+" complete.");
        }
        System.out.println("Finished"); 
        
        // save the trained weights to a new file weights.txt (overwrite if already exists)
        PrintWriter writer;
        try{
            writer = new PrintWriter("weights.txt", "UTF-8");
            for (int i = 1; i < NUM_HID_NODES; i++) {
                for (int j = 0; j < NUM_IN_NODES; j++){
                    writer.println(T.net.getLayerAt(HIDDEN_LAYER).getNodeAt(i).getWeightAt(j));
                }
            }
            for (int i = 0; i < NUM_OUT_NODES; i++) {
                for (int j = 0; j < NUM_HID_NODES; j++){
                    writer.println(T.net.getLayerAt(OUTPUT_LAYER).getNodeAt(i).getWeightAt(j));
                }
            }
            writer.close();
        }catch(IOException e){}
      
    }    
}
