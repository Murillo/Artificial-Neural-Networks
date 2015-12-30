package artificialneuralnetwork;

import java.util.Random;

public class Perceptron {
    private double[] weights;
    private double learningRate;
    private double error;
    private int input;
    private boolean training;
    public int iteration;
   
    public Perceptron(int input, double learning, double error, boolean training)
    {
        this.learningRate = learning;
        this.error = error;
        this.training = training;
        this.input = input;
        loadWeights();
    }

    private void loadWeights()
    {
        Random r = new Random();
        weights = new double[input + 1];
        for (int i = 0; i < input + 1; i++){
            weights[i] = r.nextDouble();
        }
    }

    public void startTraining(double[][] input, int[] output)
    {
        if (!training)
        {
            while (error != 0)
            {
                error = 0;
                for (int i = 0; i < output.length; i++)
                {
                    double[] ipt = new double[input[0].length];
                    for (int k = 0; k < input[0].length; k++){
                        ipt[k] = input[i][k];
                    }

                    int outputTraining = calculateOutput(ipt, weights);

                    // Calculating error
                    double localError = output[i] - outputTraining;
                    if (localError != 0)
                    {
                        // Updating the weights
                        for (int k = 0; k < weights.length; k++)
                        {
                            if (k == (weights.length - 1)){
                                weights[k] += learningRate * localError * 1;
                            }else{
                                weights[k] += learningRate * localError * input[i][k];
                            }
                        }
                    }
                    error += Math.abs(localError);
                }
                iteration++;
            }
            training = true;
        }
    }

    public int run(double[][] input)
    {
        double[] ipt = new double[input[0].length];
        for (int k = 0; k < input[0].length; k++){
            ipt[k] = input[0][k];
        }

        return calculateOutput(ipt, weights);
    }

    private int calculateOutput(double[] input, double[] weights)
    {
        double sum = 0f;
        for (int i = 0; i < weights.length; i++)
        {
            if (i == (weights.length - 1))
                sum += 1 * weights[i];
            else
                sum += input[i] * weights[i];
        }
        return (sum >= 0) ? 1 : 0;
    }
    
    public int getIteration(){
        return iteration;
    }
}
