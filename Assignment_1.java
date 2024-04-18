/*  Amani Cheatham
 *  102-81-556
 *  CSC 475 - Assignemnt 2
 *  Due: October 27, 2022
 *  
 *  This program generates a neural network AI that goes through the MNIST training data and correctly guesses what the outputs
 *  are by using Schocastic Gradient Descent and backpropogation. The MNIST training data is setup as a CSV file that this program
 *  reads from and converts it into a single array. We then go through all the inputs of the CSV using SGD and backpropogation
 *  to update the randomly generated weights and biases in hopes that it will improve in accuracy overtime. Finally, it will
 *  go through the MNIST testing set to see how much it actually learned.x`
 */


import java.util.*;
import java.io.*;
import java.text.*;

// Main Class - To instantiate the neural network and print outputs
public class Assignment_1 {
    static NeuralNetwork network = new NeuralNetwork();

    public static void startTraining()
    {
         // Instantiate the neural network - reset neural network
         network.resetNetwork();
 
         // Initialize the random weights and biases
         network.initializeNetwork();
 
         // Train the network based on the number of epochs in the network
         for(int count = 1; count <= network.epochs; count++)
         {
             // Actually trains the network.
             network.trainNetwork();
 
             // Print the accuracy for training batch
             network.printAccuracy(count, true);
         }
         // Prints the final training accuracy at the end
         network.showTrainingAccuracy();
         network.printAccuracy(0, true);
    }

    // startTesting - Prints the testing accuracy as they go through
    public static void startTesting()
    {
        // Run the network on the testing Data
        network.testNetwork();

        // Print the testing accuracy 
        network.printAccuracy(0, false);
    }

    // checkTrainingAccuracy - checks the training accuracy. Runs through the training data and prints accuracy
    public static void checkTrainingAccuracy()
    {
        // Runs through the training 
        network.showTrainingAccuracy();

        // Prints the accuracy of the training data
        network.printAccuracy(0, true);
    }
    
    // checkTestingAccuracy - checks the testing accuracy. Runs through the testing data and prints accuracy
    public static void checkTestingAccuracy()
    {
        // Runs through the testing data
        network.testNetwork();

        // Prints the accuracy of the testing data
        network.printAccuracy(0, false);
    }

    // saveNetworkToFile - saves the weights and biases to a file. Each layer of weights and biases has their own file.
    public static void saveNetworkToFile()
    {
        try
        {
            // Create a buffered writer that will create the file for L1_WEIGHTS.
            BufferedWriter writer = new BufferedWriter(new FileWriter("L1_WEIGHTS.txt"));

            // Loops through the L1_WEIGHTS and writes it to the file
            for(int i = 0; i < network.L1_WEIGHTS.getRow(); i++)
            {
                for(int j = 0; j < network.L1_WEIGHTS.getCol(); j++)
                {
                    writer.write(Double.toString(network.L1_WEIGHTS.getValue(i, j)));
                    writer.newLine();
                }
            }
            writer.close();

            // Creates a buffered writer that will create the file for the L2_WEIGHTS
            BufferedWriter writer2 = new BufferedWriter(new FileWriter("L2_WEIGHTS.txt"));
            for(int i = 0; i < network.L2_WEIGHTS.getRow(); i++)
            {
                for(int j = 0; j < network.L2_WEIGHTS.getCol(); j++)
                {
                    writer2.write(Double.toString(network.L2_WEIGHTS.getValue(i, j)));
                    writer2.newLine();
                }
            }
            writer2.close();

            // Creates a buffered writer that will create the file for the L1_BIASES
            BufferedWriter writer3 = new BufferedWriter(new FileWriter("L1_BIASES.txt"));
            for(int i = 0; i < network.L1_BIASES.getCol(); i++)
            {
                writer3.write(Double.toString(network.L1_BIASES.getValue(i)));
                writer3.newLine();
            }
            writer3.close();

            // Creates a vbuffered writer that will create the file for L2_BIASES
            BufferedWriter writer4 = new BufferedWriter(new FileWriter("L2_BIASES.txt"));
            for(int i = 0; i < network.L2_BIASES.getCol(); i++)
            {
                writer4.write(Double.toString(network.L2_BIASES.getValue(i)));
                writer4.newLine();
            }
            writer4.close();

        }

        // Incase we couldnt create any files, we catch the exception
        catch (IOException e)
        {
            System.out.println("Didnt work");
            e.printStackTrace();
        }
    }

    // loadNetworkFromFile - loads the weights and biases into the network
    public static void loadNetworkFromFile()
    {
        try
        {
            // Uses a scanner to scan each individual file and append it to each networks specific list.
            // We scan the L1_Weights file and append each value to the weights
            Scanner s1 = new Scanner(new File("L1_WEIGHTS.txt"));
            while(s1.hasNextLine())
            {
                for(int i = 0; i < network.L1_WEIGHTS.getRow(); i++)
                {
                    for(int j = 0; j < network.L1_WEIGHTS.getCol(); j++)
                    {
                        network.L1_WEIGHTS.updateValue(i, j, Double.parseDouble(s1.nextLine()));
                    }
                }
            }

            // We scan the L2_WEIGHTS and append each value to the weights.
            Scanner s2 = new Scanner(new File("L2_WEIGHTS.txt"));
            while(s2.hasNextLine())
            {
                for(int i = 0; i < network.L2_WEIGHTS.getRow(); i++)
                {
                    for(int j = 0; j < network.L2_WEIGHTS.getCol(); j++)
                    {
                        network.L2_WEIGHTS.updateValue(i, j, Double.parseDouble(s2.nextLine()));
                    }
                }
            }
            
            // We scan the L1_biases and append each value to the biases
            Scanner s3 = new Scanner(new File("L1_BIASES.txt"));
            while(s3.hasNextLine())
            {
                for(int i = 0; i < network.L1_BIASES.getCol(); i++)
                {
                    network.L1_BIASES.updateValue(i, Double.parseDouble(s3.nextLine()));
                }
                
            }

            // We scan the L2_Biases and append each value to the baises
            Scanner s4 = new Scanner(new File("L2_BIASES.txt"));
            while(s4.hasNextLine())
            {
                for(int i = 0; i < network.L2_BIASES.getCol(); i++)
                {
                    network.L2_BIASES.updateValue(i, Double.parseDouble(s4.nextLine()));
                }
            }
        }

        // Incase the file does nto exist.
        catch (FileNotFoundException e)
        {
            e.printStackTrace();
        }
    }
    
    // viewableOptions - Prints the gui in the terminal
    public static void viewableOptions(boolean trained)
    {
        // If its trained, we show all sections, otherwise just 1 or 2.
        if(trained)
        {
            System.out.println("\t\t[1]: Train your network");
            System.out.println("\t\t[2]: Load a pre-trained network");
            System.out.println("\t\t[3]: Display Network Accuracy on TRAINING data set");
            System.out.println("\t\t[4]: Display Network Accuracy on TESTING data set");
            System.out.println("\t\t[5]: Save the network state to a file");
            System.out.println("\t\t[0]: Exit");
        }
        else
        {
            System.out.println("\t\t[1]: Train your network");
            System.out.println("\t\t[2]: Load a pre-trained network");
            System.out.println("\t\t[0]: Exit");
            
        }

    }
    public static void main(String[] args)
    {
        // VARIABLES
        Console console = System.console();
        boolean isTrained = false;
        boolean tested = false;

        // Read from the csv file
        network.readDataLineByLine("mnist_train.csv", true);

        // Reads the testing file line by line.
        network.readDataLineByLine("mnist_test.csv", false);

        // Generate the random index array
        network.generateIndexArray();

        // We check to see if a console exits, if it doesn't, we return;
        if(console == null)
        {
            return;
        }
        
        // While true, we run until we quit.
        while(true)
        {
            // WE first print the logo and viewable options
            logo();
            viewableOptions(isTrained);

            // We then create user input and output.
            String output = console.readLine();
            int outputInt = Integer.parseInt(output);
            switch (outputInt)
            {
                // Switch cases based on what we need.
                case 0:
                    System.out.println("Quitting Amani's AI. GOODBYE!");
                    System.exit(0);
                    break;
                case 1:
                    startTraining();
                    isTrained = true;
                    break;
                case 2:
                    if(isTrained)
                    {
                        System.out.println("\t\tWeights and biases have already been loaded! Please use a different command");
                        continue;
                    }
                    else
                    {
                        loadNetworkFromFile();
                        System.out.println("\t\tWeights and Biases have been loaded.");
                        isTrained = true;
                    }
                    break;
                case 3:
                    checkTrainingAccuracy();
                    break;
                case 4:
                    if(tested)
                    {
                        checkTestingAccuracy();
                    }
                    else
                    {
                        startTesting();
                        tested = true;
                    }
                    break;
                case 5:
                    saveNetworkToFile();
                    break;
            }
        }
    }

    // logo - My AI's Logo. Just for aesethetics.
    public static void logo()
    {
        System.out.println();
        System.out.println("\t\t\t******************************************************************************");
        System.out.println("\t\t\t***********************     ****************                  ****************");
        System.out.println("\t\t\t**********************       ***************                  ****************");
        System.out.println("\t\t\t*********************         ********************     ***********************");
        System.out.println("\t\t\t********************  *******  *******************     ***********************");
        System.out.println("\t\t\t*******************   *******   ******************     ***********************");
        System.out.println("\t\t\t******************    *******    *****************     ***********************");
        System.out.println("\t\t\t*****************                 ****************     ***********************");
        System.out.println("\t\t\t****************                   ***************     ***********************");
        System.out.println("\t\t\t***************     **********      **************     ***********************");
        System.out.println("\t\t\t**************     ************      *************     ***********************");
        System.out.println("\t\t\t*************     **************      ************     ***********************");
        System.out.println("\t\t\t************     ****************      *****                  ****************");
        System.out.println("\t\t\t***********     ******************      ****                  ****************");
        System.out.println("\t\t\t******************************************************************************");

        System.out.println();
        System.out.println("\t\t\t Welcome to Amani's AI \t\t\t");
        System.out.println();

    }
}


// Neural Network - Class that creates the neural network.
class NeuralNetwork 
{

    // VARIABLES 
    Double learning_rate = 3.0d;
    int epochs = 30;
    int mini_batch_size = 10;
    int total_outputs = 0;
    int correct_outputs = 0;  
    int test_data_outputs = 0;
    ArrayList<Integer> INDEX_ARRAY = new ArrayList<>();
    ArrayList<ArrayList<Integer>> ALL_OUTPUTS = new ArrayList<>();
    ArrayList<ArrayList<Integer>> TEST_DATA = new ArrayList<>();
    int[][] individual_values = new int[10][2];

    // MATRIX VARIABLES
    NeuronMatrix_1d L1_ACTIVATION = new NeuronMatrix_1d(100);
    NeuronMatrix_1d L2_ACTIVATION = new NeuronMatrix_1d(10);

    NeuronMatrix_2d L1_WEIGHTS = new NeuronMatrix_2d(100, 784);
    NeuronMatrix_2d L2_WEIGHTS = new NeuronMatrix_2d(10, 100);

    NeuronMatrix_1d L1_BIASES = new NeuronMatrix_1d(100);
    NeuronMatrix_1d L2_BIASES = new NeuronMatrix_1d(10);

    NeuronMatrix_2d L1_WEIGHT_GRADIENT_SUMS = new NeuronMatrix_2d(100, 784);
    NeuronMatrix_2d L2_WEIGHT_GRADIENT_SUMS = new NeuronMatrix_2d(10, 100);
    
    NeuronMatrix_1d L1_BIAS_GRADIENT_SUMS = new NeuronMatrix_1d(100);
    NeuronMatrix_1d L2_BIAS_GRADIENT_SUMS = new NeuronMatrix_1d(10);
    NeuronMatrix_1d L2_BIAS_GRADIENT = new NeuronMatrix_1d(10);
    NeuronMatrix_1d L1_BIAS_GRADIENT = new NeuronMatrix_1d(100);

    NeuronMatrix_1d NEURON_ACTIVATION = new NeuronMatrix_1d(784);
    NeuronMatrix_1d ONE_HOT_VECTOR = new NeuronMatrix_1d(10);


    // readDataLineByLine - reads from the CSV file. It will go line by line and add the values of the csv file to 2d ArrayList
    // training - determine if it is training or not.
    public void readDataLineByLine(String csvFile, boolean training)
    {
        // Create a string to parse the CSV file by
        final String delimiter = ",";
        try
        {
            // We create a java File from the input of the functio
            File file = new File(csvFile);

            // We create a java Filereader that reads the java File
            FileReader filereader = new FileReader(file);

            // We create a java BufferedReader that reads the file reader.
            BufferedReader br = new BufferedReader(filereader);

            // Create a string that will read from the buffered reader
            String line = "";

            // Create a string array that will split the file into indiviual parts and put them in the array
            String[] tempArr;
            while((line = br.readLine()) != null)
            {
                // We split the array
                tempArr = line.split(delimiter);

                // Create an ArrayList of Integers to store data in. We will convert it to an Integer and add it to the array
                ArrayList<Integer> tempList = new ArrayList<Integer>();
                for(String tempStr : tempArr)
                {
                    tempList.add(Integer.parseInt(tempStr));
                }
                
                // We check to see if its training - if it is, we add the list to all_outputs list, if it isnt, we add it to the test_data list
                if(training)
                {
                    // Adds the list to ALL_OUPUTS in the neural network
                    ALL_OUTPUTS.add(tempList);
    
                    // Increase the total outputs count;
                    total_outputs += 1;
                }
                else
                {
                    // Adds the list to testData
                    TEST_DATA.add(tempList);

                    // Increases the
                    test_data_outputs += 1;
                }
            }
            
            // We close the buffered reader
            br.close();
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
    }

    /*  
     *  FeedForward - The function that goes forward through the network.
     *  Update_Activations - A NeuronMatrix_1d that are the activations that will be updated by going through the network. This will be the L+1 activations
     *  Weights - A NeuronMatrix_2d that are the weights associated with connecting this layer to the L+1 layer
     *  Activations - A NeuronMatrix_1d that are the activations of this layers nodes.
     *  Biases - A NeuronMatrix_1d that are the biases associated with the L+1 layers nodes.
     */
    public void FeedFoward(NeuronMatrix_1d update_activations, NeuronMatrix_2d weights, NeuronMatrix_1d activations, NeuronMatrix_1d biases)
    {
        // We go through the row of the Weights
        for(int i = 0; i < weights.getRow(); i++)
        {
            // We create temporary variables to store the values we will get
            double tempSum = 0.0d;
            double finalSum = 0.d;

            // We loop through the column and do the dot product of the weights at row i column j and multiply them by the activation at column j
            // We can do this since the weights col and activations column are the same
            for(int j = 0; j < weights.getCol(); j++)
            {
                tempSum += weights.getValue(i, j) * activations.getValue(j);
            }

            // After the dot product we add the bias at that node to the sum and do the sigmoid function on it. We then update
            // the next layers activation with that value.

            tempSum += biases.getValue(i);
            finalSum = 1/(1+Math.exp(-tempSum));
            update_activations.updateValue(i, finalSum);
        }

    }

    /*
     * backPropogationOutputLayer - We go backwards through the network and try to see where it went wrong. This starts from the output
     * to the hidden layers.
     * one_hot_vector - A NeuronMatrix_1d that is the matrix where the correct value is 1.0d and all others are 0.0. Ie if the answer is 5 then the array would be [0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0]
     * current_activations - a NeuronMatrix_1d that is this layers activationss. 
     * prev_activations - A NeuronMatrix_1d that has the L-1 layer activations. 
     * bias_gradient - A NM1d array that is the bias gradient for this layer.
     * sum_bias_gradient - A NM1d array that will hold the sums of all bias gradients for this layer.
     * sum_weight_gradient - NM2d arrat that will hold the sum of all weight gradients for this layer.
     */
    public void backPropogationOutputLayer(NeuronMatrix_1d one_hot_vector, NeuronMatrix_1d current_activations, NeuronMatrix_1d prev_activations, 
                                            NeuronMatrix_1d bias_gradient, NeuronMatrix_1d sum_bias_gradient, NeuronMatrix_2d sum_weight_gradient)
    {
        // We loop through the currect activations;
        for(int i = 0; i < current_activations.getCol(); i++)
        {
            // We create a tempSum to store the value and tempValue to get the bias_gradient_sum value at index i. 
            // Final sum is used to keep track of the changes to our bias gradient sum through all the mini batches.
            double tempSum = 0.0d;
            double tempValue;
            double finalSum;

            // Temp sum is set to the gradient of the output value. 
            tempSum = (current_activations.getValue(i) - one_hot_vector.getValue(i)) * current_activations.getValue(i) * (1 - current_activations.getValue(i));
            tempValue = sum_bias_gradient.getValue(i);

            // Set final sum to temp sum + tempValue to update our bias_gradient
            finalSum = tempSum + tempValue;

            // We update the bias_gradient and sum_bias gradient to the new values respectively
            bias_gradient.updateValue(i, tempSum);
            sum_bias_gradient.updateValue(i, finalSum);
     
        }

        // we loop through the previous activations 
        for(int i = 0; i < prev_activations.getCol(); i++)
        {
            // we loop through the bias gradient
            for(int j = 0; j < bias_gradient.getCol(); j++)
            {
                // We create a tempSum to keep track of the values, tempValue will store the previous value
                // final Sum will add the temp value and the previous sum to the sum_weight_gradient for this layer
                double tempSum = 0.0d;
                double tempValue;
                double finalSum;
                
                // We set tempSum to the bias_gradient and the jth value * previous activation value and the ith value
                // We set tempValue equal to the previous sum_weight gradient_value to add to this new tempSum
                // We update the sum_weight_gradient and at the jth row and ith column
                tempSum = bias_gradient.getValue(j) * prev_activations.getValue(i);
                tempValue = sum_weight_gradient.getValue(j, i);
                finalSum = tempSum + tempValue;
                sum_weight_gradient.updateValue(j, i, finalSum);
            }
        }
    }
    /*
     * backPropogationHiddenLayer - We go backwards through the hidden layers of the neural network.
     * Weights - A NM2d that connects this layer to the l+1 layer.
     * previous_activations - A NM1d that is the l-1 layer activations
     * self_activations - A NM1d that is this layers activation rates
     * prev_bias_gradient - the bias graident of the l+1 layer
     * self_bias_gradient - The bias gradient of this layer.
     * sum_bias_gradient - the summation of bias gradients that correspond to this layer in the neural network
     * sum_weight_gradient - the summation of the weight gradients that correspond to this layer in the neural network
     */
    public void backPropogationHiddenLayer(NeuronMatrix_2d weights, NeuronMatrix_1d previous_activations, 
                                            NeuronMatrix_1d self_activations, NeuronMatrix_1d prev_bias_gradient, NeuronMatrix_1d self_bias_gradient,
                                            NeuronMatrix_1d sum_bias_gradient, NeuronMatrix_2d sum_weight_gradient)
    {
        // We go through this layers bias gradients and update the bias gradient and sum bias gradient
        for(int i = 0; i < self_bias_gradient.getCol(); i++)
        {
            // We create a tempSum to store the sum of the weights dot product the prev_bias_gradient
            // Temp value to store the previous sum value at i index
            // tempSum2 to calculate the gradient of the the summation of tempSum
            // Final sum to update the sum_bias_gradients
            double tempSum = 0.0d;
            double tempValue;
            double tempSum2 = 0.0d;
            double finalSum;    

            for(int j = 0; j < weights.getRow(); j++)
            {
                // We get the value at the jth row ith column of weights and multiply times the prev_bias_gradient at jth column
                // We add on this to get the sum of this neuron activation.
                tempSum += weights.getValue(j, i) * prev_bias_gradient.getValue(j);
            }

            // We set tempValue to this layers activation rate
            tempValue = self_activations.getValue(i);
            
            // We set tempSum2 equal to the gradient by doing the math
            tempSum2 = tempSum * (tempValue * (1 - tempValue));

            // We set finalSum to tempSum2
            finalSum = tempSum2;

            // We add it the previous sum_bias_gradient and the ith col.
            finalSum += sum_bias_gradient.getValue(i);

            // We set self_bias_gradent to tempsum2 and ith col and sum_bias_gradient to final sum at ith column
            self_bias_gradient.updateValue(i, tempSum2);
            sum_bias_gradient.updateValue(i, finalSum);
        }

        // We go through this previous activations col to get the columns at which each activation is associated with
        for(int i = 0; i < previous_activations.getCol(); i++)
        {
            // We go through the bias gradient of this layer to get the change we need to make to the weights
            for(int j = 0; j < self_bias_gradient.getCol(); j++)
            {
                // We create a tempSum to store the values
                // tempValue to get the previous sums value
                // finalSum to combine the two values
                double tempSum = 0.0d;
                double tempValue;
                double finalSum;
                tempSum = self_bias_gradient.getValue(j) * previous_activations.getValue(i);
                tempValue = sum_weight_gradient.getValue(j,i);
                finalSum = tempSum + tempValue;
                // We then update sum_weight_gradient at the jth row ith column to its new value
                sum_weight_gradient.updateValue(j, i, finalSum);
            }
        }
    }

    // generateNeuronActivation - This creates the the Neuron Activation to start the FeedForward or learning process
    // Inputs - An ArrayList of integers that is the correct answer (label) and values from a line in the CSV file.
    public void generateNeuronActivation(ArrayList<Integer> inputs)
    {
        // We go through each individual element in the inputs list
        for(int i = 0; i < inputs.size(); i++)
        {
            // We create an int temp value to store the value from the inputs list
            int tempValue;

            // We create a double tempScaledValue to store the double of the tempvalue
            double tempScaledValue = 0.0d;

            // If its the first element in the array, we create the one hot vector on it, as it is the correct answer for this single input
            if(i == 0)
            {
                // Generates the one hot vector
                ONE_HOT_VECTOR.one_hot_vector_creation(inputs.get(i));
            }
            else
            {
                // We set tempValue to the value at input i.
                tempValue = inputs.get(i);
                
                // We set scaled tempValue to tempValue/255.0d to convert it to a double and scale the range from 0-1.
                tempScaledValue = tempValue/255.0d;

                // We then set the neuron activation at the i-1 value to the tempScaledValue
                // We do i-1 because i will always start at 1 due to the first item being the label.
                // So we subtract 1 to start and 0 and stay in the NEURON_ACTIVATION length/
                NEURON_ACTIVATION.updateValue(i-1, tempScaledValue);
            }
        }
    }

    // resetNetwork - resets everything incase we want to train the network again or more than once.
    public void resetNetwork()
    {
        clearActivations();
        clearGradients();
        reinstantiateActivations();
        reinstantiateSums();
        correct_outputs = 0;
    }
    /*
     * UpdateWeightsAndBiases - We update the weights and biases
     * Weights - The weights associated with the layer we want to update the weights for
     * Sum_Weight_Gradients - The sum of the weight gradients to represent how much to change the weight by
     * Biases - The biases associated with the layer we want to update the biases for
     * sum_bias_gradient - the sum of the biases to represent how much to change the weight by
     * learning_rate - the eta or the training number.
     * mini_batch_size - the size of the mini_batch
     */

    public void UpdateWeightsAndBiases(NeuronMatrix_2d weights, NeuronMatrix_2d sum_weight_gradient, NeuronMatrix_1d biases, 
                                        NeuronMatrix_1d sum_bias_gradient, double learning_rate, int mini_batch_size)
    {
        // We go through the row of weights
        for(int i = 0; i < weights.getRow(); i++)
        {
            // we go through the column of weights
            for(int j = 0; j < weights.getCol(); j++)
            {

                // We create a temp value to store the math behind updating the weights
                // listValue 1 to store the value at the weights
                // listValue 2 to store teh value at the sum_of_weights
                double tempValue = 0.0d;
                double listValue1;
                double listValue2;
                listValue1 = weights.getValue(i, j);
                listValue2 = sum_weight_gradient.getValue(i, j);

                // We do the math to calculate what the weight should be.
                tempValue = listValue1 - (learning_rate/Double.valueOf(mini_batch_size)) * listValue2;

                // We then update the weights at the ith row jth column to tempValue.
                weights.updateValue(i, j, tempValue);
            }
        }

        // We go through the column of biases
        for(int i = 0; i < biases.getCol(); i++)
        {

            // Tempvalue to store the math behind updating the biases
            // listValue1 to store the value at the biases
            // listValue2 to store the value at the sum_biases
            double tempValue = 0.0d;
            double listValue1;
            double listValue2;
            listValue1 = biases.getValue(i);
            listValue2 = sum_bias_gradient.getValue(i);

            // We do the math to calculate what the bias should be
            tempValue = listValue1 - (learning_rate/Double.valueOf(mini_batch_size)) * listValue2;

            // We update the bias at ith column to tempValue
            biases.updateValue(i, tempValue);
        }
    }

    // Initialize Network - We initialize the network by generating random weights and biases associated with each layer in the network
    // These weights and biases are random values between -1 and 1.
    public void initializeNetwork()
    {
        L1_BIASES.generateRandomBiases(-1, 1);
        L2_BIASES.generateRandomBiases(-1, 1);
        L1_WEIGHTS.generateRandomWeights(-1, 1);
        L2_WEIGHTS.generateRandomWeights(-1, 1);
    }

    // clearActivations - we clear all the activations for the network
    public void clearActivations()
    {
        L1_ACTIVATION.clearList();
        NEURON_ACTIVATION.clearList();
        L2_ACTIVATION.clearList();
        ONE_HOT_VECTOR.clearList();
    }

    // reinstaniateActivations - we recreate or reinsantiate the activations
    public void reinstantiateActivations()
    {
        L1_ACTIVATION.reinstate();
        NEURON_ACTIVATION.reinstate();
        L2_ACTIVATION.reinstate();
        ONE_HOT_VECTOR.reinstate();
    }

    // clearGradients - we clear all of the gradients for the network
    public void clearGradients()
    {
        L1_WEIGHT_GRADIENT_SUMS.clearList();
        L2_WEIGHT_GRADIENT_SUMS.clearList();
        L1_BIAS_GRADIENT_SUMS.clearList();
        L2_BIAS_GRADIENT_SUMS.clearList();
        L1_BIAS_GRADIENT.clearList();
        L2_BIAS_GRADIENT.clearList();
    }

    // reinstantiateSums - we recreate or reinstantiate the gradients
    public void reinstantiateSums()
    {
        L1_WEIGHT_GRADIENT_SUMS.reinstate();
        L2_WEIGHT_GRADIENT_SUMS.reinstate();
        L1_BIAS_GRADIENT_SUMS.reinstate();
        L2_BIAS_GRADIENT_SUMS.reinstate();
        L1_BIAS_GRADIENT.reinstate();
        L2_BIAS_GRADIENT.reinstate();
    }

    // resetCorrecOutputs - we reset the correctOutputs and individualvalues
    public void resetCorrectOutputs()
    {
        this.correct_outputs = 0;
        for(int i = 0; i < 10; i++)
        {
            for(int j = 0; j < 2; j++)
            {
                individual_values[i][j] = 0;
            }
        }
    }

    // addCorrectValues - we add the correct values to their respective list and increment the correct_outputs
    // index - the index of where the correct value is located. Will be a value between 0-9
    public void addCorrectValues(int index)
    {  
        // a switch case for whatever value of index is
        switch (index)
        {
            case 0:
                individual_values[0][0] += 1;
                individual_values[0][1] += 1;
                correct_outputs += 1;
                break;
            case 1:
                individual_values[1][0] += 1;
                individual_values[1][1] += 1;
                correct_outputs += 1;
                break;
            case 2:
                individual_values[2][0] += 1;
                individual_values[2][1] += 1;
                correct_outputs += 1;
                break;
            case 3:
                individual_values[3][0] += 1;
                individual_values[3][1] += 1;
                correct_outputs += 1;
                break;
            case 4:
                individual_values[4][0] += 1;
                individual_values[4][1] += 1;
                correct_outputs += 1;
                break;
            case 5:
                individual_values[5][0] += 1;
                individual_values[5][1] += 1;
                correct_outputs += 1;
                break;
            case 6:
                individual_values[6][0] += 1;
                individual_values[6][1] += 1;
                correct_outputs += 1;
                break;
            case 7:
                individual_values[7][0] += 1;
                individual_values[7][1] += 1;
                correct_outputs += 1;
                break;
            case 8:
                individual_values[8][0] += 1;
                individual_values[8][1] += 1;
                correct_outputs += 1;
                break;
            case 9:
                individual_values[9][0] += 1;
                individual_values[9][1] += 1;
                correct_outputs += 1;
                break;
        }
    }

    // addIncorrectValues - it adds the incorrect value to the corresponding value
    // index - the value at which the CORRECT output should have been;
    public void addIncorrectValues(int index)
    {
        // A switch case for each value of the index
        switch (index)
        {
            case 0:
                individual_values[0][1] += 1;
                break;
            case 1:
                individual_values[1][1] += 1;
                break;
            case 2:
                individual_values[2][1] += 1;
                break;
            case 3:
                individual_values[3][1] += 1;
                break;
            case 4:
                individual_values[4][1] += 1;
                break;
            case 5:
                individual_values[5][1] += 1;
                break;
            case 6:
                individual_values[6][1] += 1;
                break;
            case 7:
                individual_values[7][1] += 1;
                break;
            case 8:
                individual_values[8][1] += 1;
                break;
            case 9:
                individual_values[9][1] += 1;
                break;
        }
    }

    // checkForHighestValue - Goes through the input and checks for the highestValue
    // input - the output list to check for which index has the highestValue
    // one_hot_vector - the list that is the matrix for correct output
    public void checkForHighestValue(NeuronMatrix_1d input, NeuronMatrix_1d one_hot_vector)
    {
        // We create a double that will store the highest value
        // an Integer index to store the index of the highest value
        // a correct_index Integer to store the correct_index from the one_hot_vector
        double highestValue = 0.0d;
        Integer index = 0;
        Integer correct_index = 0;

        // we Go through the inputs col
        for(int i = 0; i < input.getCol(); i++)
        {
            // We get the value at the ith column and see if its higher.
            // If it is, we update highestValue to that value and the index to i
            if(input.getValue(i) > highestValue)
            {
                highestValue = input.getValue(i);
                index = i;
            }

            // We check and see if the one hot vector at i index is 1.0d. If it is, we store it as the correct_index
            if(one_hot_vector.getValue(i) == 1.0d)
            {
                correct_index = i;
            }
        }

        // if the index matches the correct index, we update the values using addCorrectValues method
        // else we update the values using the addIncorrectValues method
        if(index == correct_index)
        {
            addCorrectValues(index);
        }
        else
        {
            addIncorrectValues(correct_index);
        }
    }

    /*
     * SchocasticGradientDescent - This is where the learning takes place. It will feed forward through the network and then
     * use backPropogation to generate the changes needed to be made to the weights and biases.
     * Input - A single list containing the label and the activations for that label.
     * mini_batch_size - the size of the mini-batches
     * learning_rate - the eta, or the rate at which it learns.
     */
    public void SchocasticGradientDescent(ArrayList<Integer> input)
    {
        // Generates neuron activation base don the input
        generateNeuronActivation(input);
        // Feeds fordward from L0 -> L1
        FeedFoward(L1_ACTIVATION, L1_WEIGHTS, NEURON_ACTIVATION, L1_BIASES);
        // Feeds Forward from L1 -> L2
        FeedFoward(L2_ACTIVATION, L2_WEIGHTS, L1_ACTIVATION, L2_BIASES);
        // Checks the highest value and updates if its correct or incorrect
        checkForHighestValue(L2_ACTIVATION, ONE_HOT_VECTOR);
        // Backprops from the outputLayer to the hidden Layer
        backPropogationOutputLayer(ONE_HOT_VECTOR, L2_ACTIVATION, L1_ACTIVATION, L2_BIAS_GRADIENT, L2_BIAS_GRADIENT_SUMS, L2_WEIGHT_GRADIENT_SUMS);
        // Backprops from through the hidden Layer
        backPropogationHiddenLayer(L2_WEIGHTS, NEURON_ACTIVATION, L1_ACTIVATION, L2_BIAS_GRADIENT, L1_BIAS_GRADIENT, L1_BIAS_GRADIENT_SUMS, L1_WEIGHT_GRADIENT_SUMS);
    }

    // generateIndexArray - Generates an Index Array based on the size of the ouputs.
    // It creates an array that will contain a list in Integers that represent the indexes of the outputs List
    public void generateIndexArray()
    {
        // We loop through the ouputs and append i to the Index_Array
        for(int i = 0; i < ALL_OUTPUTS.size(); i++)
        {
            INDEX_ARRAY.add(i);
        }
    }

    // print Accuracy - Prints to the system the information after it has run through the entire list
    // count - represents what training Epoch we are on
    // training - determines if we are training or not. Used to check what to print or not print
    public void printAccuracy(int count, boolean training)
    {
        // We create a Decimal format to round the ints to a specific decimal. In this case, we do 3 decimal places.
        DecimalFormat df = new DecimalFormat("#.###");

        // We create two seperate percents, one for training and one for the testing.
        double trainingPercent = (Double.valueOf(correct_outputs)/Double.valueOf(total_outputs)) * 100 ;
        double testingPercent = (Double.valueOf(correct_outputs)/Double.valueOf(test_data_outputs) * 100);
        String stringFormat = "| %-9s | %-9d |   %-5d   |%n";

        // If its training, we print the training set count
        // If it isnt, we print the testing set.
        if(training)
        {
            if(count == 0)
            {
                System.out.println(" ----------- FINAL RUN ------------- ");
            }
            else
            {
                System.out.println(" --------- TRAINING SET " + count + " ---------- ");
            }
        }
        else
        {
            System.out.println(" ---------- TESTING SET ------------ ");
        }

        System.out.format("+-----------------------------------+%n");
        System.out.format("|  Integer  |  Correct  |   Total   |%n");
        System.out.format("+-----------------------------------+%n");
        // We loop through the individual values and display them to the console
        for(int i = 0; i < 10; i++)
        {
            //System.out.print(i + " = " + individual_values[i][0] + "/" + individual_values[i][1] + " \t ");
            System.out.format(stringFormat, i, individual_values[i][0], individual_values[i][1]);
        }
        // If its training, it shows the accuracy of the training data for each epoch.
        // If its not, it shows the output for the test_data accuracy
        if(training)
        {
            //System.out.println("Accuracy = " + correct_outputs + "/" + total_outputs + " = " + df.format(trainingPercent) + "%");
            System.out.format(stringFormat, "Accuracy", correct_outputs, total_outputs);
            System.out.format("+-----------+-----------+-----------+%n");
            System.out.println("              " + df.format(trainingPercent) + "%            ");
        }
        else
        {
            //System.out.println("Accuracy = " + correct_outputs + "/" + test_data_outputs + " = " + df.format(testingPercent) + "%");
            System.out.format(stringFormat, "Accuracy", correct_outputs, test_data_outputs);
            System.out.format("+-----------+-----------+-----------+%n");
            System.out.println("              " + df.format(testingPercent) + "%            ");

        }
        System.out.println();
        // We reset the output each time this function is called.
        resetCorrectOutputs();
    }

    public void start_mini_batch(List<Integer> inputs, int mini_batch_size, double learning_rate)
    {
        for(int i = 0; i < inputs.size(); i++)
        {
            // It uses SchocasticGradientDescent to go through the entire data set.
            SchocasticGradientDescent(ALL_OUTPUTS.get(inputs.get(i)));
            // We clear the activations so that we can setup for the next value
            clearActivations();
            // We reinstantiate the activations
            reinstantiateActivations();
        }

        // Update weights and biases for each layer
        UpdateWeightsAndBiases(L2_WEIGHTS, L2_WEIGHT_GRADIENT_SUMS, L2_BIASES, L2_BIAS_GRADIENT_SUMS, learning_rate, mini_batch_size);
        UpdateWeightsAndBiases(L1_WEIGHTS, L1_WEIGHT_GRADIENT_SUMS, L1_BIASES, L1_BIAS_GRADIENT_SUMS, learning_rate, mini_batch_size);
        // Clear the gradients and reinstantiate them
        clearGradients();
        reinstantiateSums();
    }

    // We train the network. 
    public void trainNetwork()
    {
        // We shuffle the index array so that the list is never the same order.
        // We can tell if its learning this way.
        Collections.shuffle(INDEX_ARRAY);
        ArrayList<Integer> full_mini_batch_list = new ArrayList<>(INDEX_ARRAY);
        for(int i = 0; i < INDEX_ARRAY.size(); i += mini_batch_size)
        {
            List<Integer> small_mini_batches = full_mini_batch_list.subList(i, i+mini_batch_size);
            start_mini_batch(small_mini_batches, mini_batch_size, learning_rate);
        }
    }

    public void showTrainingAccuracy()
    {
        for(int i = 0; i < ALL_OUTPUTS.size(); i++)
        {
            generateNeuronActivation(ALL_OUTPUTS.get(i));
            // We feed forward from L0 -> L1
            FeedFoward(L1_ACTIVATION, L1_WEIGHTS, NEURON_ACTIVATION, L1_BIASES);
            // We go forward from L1 ->2
            FeedFoward(L2_ACTIVATION, L2_WEIGHTS, L1_ACTIVATION, L2_BIASES);
            // We check for the highestValue and see if it correct
            checkForHighestValue(L2_ACTIVATION, ONE_HOT_VECTOR);
            // We clear the activations
            clearActivations();
            // We reinstate the activations
            reinstantiateActivations();
        }
    }

    // testNetwork - we run the network on the testData and see the results.
    // We use the biases and weights we calculated through training it
    public void testNetwork()
    {
        // We go through the testData
        for(int i = 0; i < TEST_DATA.size(); i++)
        {
            // We generate the Neuron Activation
            generateNeuronActivation(TEST_DATA.get(i));
            // We go forward from L0 -> L1
            FeedFoward(L1_ACTIVATION, L1_WEIGHTS, NEURON_ACTIVATION, L1_BIASES);
            // We go forward from L1 ->2
            FeedFoward(L2_ACTIVATION, L2_WEIGHTS, L1_ACTIVATION, L2_BIASES);
            // We check for the highestValue and see if it correct
            checkForHighestValue(L2_ACTIVATION, ONE_HOT_VECTOR);
            // We clear the activations
            clearActivations();
            // We reinstate the activations
            reinstantiateActivations();
        }
    }
}

// Neutron Matrix - the class that will help with matrix multiplications. This is just a skeleton for the other two
class NeuronMatrix 
{
    // int row and col to define the matrix by
    int row;
    int col;
    
    // The constructor for this class, takes in a row, and col and sets it to those values.
    public NeuronMatrix(int row, int col)
    {
        this.row = row;
        this.col = col;
    }
}


// NeuronMatrix_1d that is a subclass of NeuronMatrixes. These are our 1-d arrays, to make our life a little easier
class NeuronMatrix_1d extends NeuronMatrix
{
    // Creates an ArrayList of Doubles to store our values
    ArrayList<Double> neuronMatrix_1d = new ArrayList<>();

    // The constructor. These matrices will always be a COLUMN matrix, or atleast feel like it to make multiplying matrices easier
    // It takes in a col and set the row to 1
    public NeuronMatrix_1d(int col)
    {
        // We called the super constructor to initialize the list
        super(1, col);
        // We go through the list and add 0.0d to the list to store as base values
        for(int i = 0; i < col; i++)
        {
            neuronMatrix_1d.add(0.0d);
        }
    }

    // getList - returns the list;
    public ArrayList<Double> getList()
    {
        return this.neuronMatrix_1d;
    }

    // getValue - returns a value at an index. This is the getter
    // index - the index at which we want the value
    public Double getValue(int index)
    {
        return this.neuronMatrix_1d.get(index);
    }

    // updateValue - we update the value at index with the new value
    // index - the index at which we want to update our value
    // value - the new value to update
    public void updateValue(int index, double value)
    {
        this.neuronMatrix_1d.set(index, value);
    }

    // getRow - returns the value of this.row
    public int getRow()
    {
        return this.row;
    }

    // getCol - returns the value of this.col
    public int getCol()
    {
        return this.col;
    }

    // clearList - clears the ArrayList;
    public void clearList()
    {
        this.neuronMatrix_1d.clear();
    }

    // reinsate - Recreates the arrayList based on the col value
    public void reinstate()
    {
        for(int i = 0; i < this.col; i++)
        {
            neuronMatrix_1d.add(0.0d);
        }
    }

    // Generates randomBiases for this 1d Array
    // min - the minimum number the random will go to
    // max - the maximum number the random will go to
    public void generateRandomBiases(int min, int max)
    {
        for(int i = 0; i < this.neuronMatrix_1d.size(); i++)
        {
            Random random = new Random();
            Double doubleRandom = random.nextDouble(max+1)+min;
            this.neuronMatrix_1d.set(i, doubleRandom);
        }
    }

    // one_hot_vector_creation - creates a one hot vector.
    // value - the index and which the correct value is
    public void one_hot_vector_creation(int value)
    {
        for(int i = 0; i < this.col; i++)
        {
            if(i == value)
            {
                this.neuronMatrix_1d.set(i, 1.0d);
            }
            else
            {
                this.neuronMatrix_1d.set(i, 0.0d);
            }
        }
    }
}

// NeuronMatrix_2d class that inherits from NeuronMatrix
class NeuronMatrix_2d extends NeuronMatrix
{
    // Creates an ArrayList<ArrayList<Doubles>> or a 2d arrayList
    ArrayList<ArrayList<Double>> neuronMatrix_2d = new ArrayList<>();

    // constructor - takes row and col and sets it to those values
    public NeuronMatrix_2d(int row, int col)
    {
        // calls the super class and creates the row, col
        super(row, col);
        // loops through the rows and creates a new arrayList
        for(int i = 0; i < row; i++)
        {
            this. neuronMatrix_2d.add(new ArrayList<>());
            // Goes through the column and add 0.0 as base values into the ith array
            for(int j = 0; j < col; j++)
            {
                this.neuronMatrix_2d.get(i).add(0.0d);
            }
        }
    }

    // getList - returns the 2d matrix
    public ArrayList<ArrayList<Double>> getList()
    {
        return this.neuronMatrix_2d;
    }

    // getValue - returns the value at row x column y
    // row - the value at which we want the sub array
    // col - the value at which we want the value inside the sub array
    public Double getValue(int row, int col)
    {
        return this.neuronMatrix_2d.get(row).get(col);
    }

    // updateValue - updates the value at row x column y
    // row - the value at which we want the sub array
    // col - the value at which we want the valueOf
    // value - the value we want to update.
    public void updateValue(int row, int col, double value)
    {
        this.neuronMatrix_2d.get(row).set(col, value);
    }

    // getRow - returns the row
    public int getRow()
    {
        return this.row;
    }

    // getCol - returns the col
    public int getCol()
    {
        return this.col;
    }

    // clearList - clears the list.
    public void clearList()
    {
        this.neuronMatrix_2d.clear();
    }

    // reinstate - recreates the 2d matrix
    public void reinstate()
    {
        for(int i = 0; i < this.row; i++)
        {
            this.neuronMatrix_2d.add(new ArrayList<>());
            for(int j = 0; j < this.col; j++)
            {
                this.neuronMatrix_2d.get(i).add(0.0d);
            }
        }
    }

    // generateRandomWeights - generates randomWeights for our network
    // min - the lowest number to be randomly generated
    // max - the highest number to be randomly generated
    public void generateRandomWeights(int min, int max)
    {
        for(int i = 0; i < this.row; i++)
        {
            int length = this.col;
            for(int j = 0; j < length; j++)
            {
                Random random = new Random();
                Double doubleRandom = random.nextDouble(max+1)+min;
                this.neuronMatrix_2d.get(i).set(j, doubleRandom);
            }
        }
    }

}