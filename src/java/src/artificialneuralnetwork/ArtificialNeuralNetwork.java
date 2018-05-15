package artificialneuralnetwork;

import artificialneuralnetwork.perceptron.Perceptron;
import artificialneuralnetwork.mlp.MultilayerPerceptron;

public class ArtificialNeuralNetwork {
    
    public static void main(String[] args) {
        
        double[][] inputAnd = new double[][] { { 1, 0 }, { 1, 1 }, { 0, 1 }, { 0, 0 } };
        int[] outputAnd = new int[] { 0, 1, 0, 0 };

        Perceptron p1 = new Perceptron(2, 0.1, 0.1, false);
        p1.startTraining(inputAnd, outputAnd);
        System.out.println("Artificial Neural Network - Perceptron");
        System.out.println("AND Gate");
        System.out.println("Iteration of training: " + p1.getIteration());
        System.out.println("Test 1: " + p1.run(new double[][] { { 1, 0 } }));
        System.out.println("Test 2: " + p1.run(new double[][] { { 1, 1 } }));
        
        double[][] inputOr = new double[][] { { 1, 0 }, { 1, 1 }, { 0, 1 }, { 0, 0 } };
        int[] outputOr = new int[] { 1, 1, 1, 0 };
        Perceptron p2 = new Perceptron(2, 0.1, 0.1, false);
        p2.startTraining(inputOr, outputOr);
        System.out.println("OR Gate");
        System.out.println("Iteration of training: " + p2.getIteration());
        System.out.println("Test 1: " + p2.run(new double[][] { { 0, 1 } }));
        System.out.println("Test 2: " + p2.run(new double[][] { { 0, 0 } }));
        System.out.println("Test 3: " + p2.run(new double[][] { { 1, 1 } }));
        
        double[][] inputNot = new double[][] { { 1 }, { 0 } };
        int[] outputNot = new int[] { 0, 1 };
        Perceptron p3 = new Perceptron(1, 0.1, 0.1, false);
        p3.startTraining(inputNot, outputNot);
        System.out.println("NOT Gate");
        System.out.println("Iteration of training: " + p3.getIteration());
        System.out.println("Test 1: " + p3.run(new double[][] { { 0 } }));
        System.out.println("Test 2: " + p3.run(new double[][] { { 1 } }));
        

        System.out.println("----------------------");
        System.out.println("Artificial Neural Network - MLP");
        MultilayerPerceptron mlp = new MultilayerPerceptron(2, 6, 1);
        mlp.training(new double[][] { { 1, 1 }, { 1, 0 }, { 0, 0 }, { 0, 1 } }, new double[] { 1, 0, 0, 0 });
        System.out.println("AND Gate: " + Math.round(mlp.run(new double[] { 1, 1 })[0]));
    }
    
}
