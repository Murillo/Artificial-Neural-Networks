package artificialneuralnetwork.mlp;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;


public class Neuron
{
    private List<Integer> code;
    private String name;
    private Double output;
    private List<Double> weight;
    private Double gradient;
    private final Double euler = 2.718281828;
    private final Double learningRate = 0.5;

    public Neuron(int inputs, Random rnd) {
        this.name = java.util.UUID.randomUUID().toString();
        this.code = new ArrayList<Integer>();
        this.weight = new ArrayList<Double>();
        for (int i = 0; i < inputs; i++){
            this.weight.add(rnd.nextDouble() * 2 - 1);
        }
    }

    public double activate(double result) {
        this.output = (1 / (1 + Math.pow(this.euler, -result)));
        return this.output;
    }

    public Double loadNeuron(List<Double> input) {
        double value = 0;
        for (int i = 0; i < this.weight.size(); i++) {
            value += input.get(i) * this.weight.get(i);
        }
        return value;
    }

    public void updateWeightsOutputLayer(Double output, List<Double> outputLeft, Double target) {
        this.gradient = output * ((1 - output) * (target - output));
        for (int i = 0; i < this.weight.size(); i++)
            this.weight.set(i, this.weight.get(i) + this.learningRate * this.gradient * outputLeft.get(i));
    }

    public void updateWeightsHiddenLayer(Double output, List<Double> outputLeft, List<Double> weightRight, List<Double> gradientRight) {
        double valueRight = 0;
        for (int i = 0; i < weightRight.size(); i++) 
            valueRight += gradientRight.get(i) * this.weight.get(i);
        
        this.gradient = output * ((1 - output) * valueRight);
        for (int i = 0; i <  this.weight.size(); i++)
            this.weight.set(i, this.weight.get(i) + this.learningRate * this.gradient * outputLeft.get(i));
    }
    
    public Double getOutput() {
        return this.output;
    }
 
    public Double getGradient() {
        return this.gradient;
    }
    
    public Double getWeight(Integer index) {
        return this.weight.get(index);
    }
    
}
