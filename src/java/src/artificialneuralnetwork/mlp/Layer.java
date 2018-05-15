package artificialneuralnetwork.mlp;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Layer
{
    public List<Neuron> neurons;
    public String name;

    public Layer(int inputs, int neurons)
    {
        Random random = new Random();
        this.neurons = new ArrayList<Neuron>();
        for (int i = 0; i < neurons; i++)
            this.neurons.add(new Neuron(inputs, random));
    }
    
    public int getNeuronsSize(){
        return this.neurons.size();
    }
    public Neuron getNeuron(int index) {
        return this.neurons.get(index);
    }
    
    public List<Double> getOutputs() {
        List<Double> weights = new ArrayList<>();
        for (int i = 0; i < neurons.size(); i++) {
            weights.add(neurons.get(i).getOutput());
        }
        return weights;
    }
    
    public List<Double> getWeights(Integer indexNeuton) {
        List<Double> weights = new ArrayList<>();
        for (int i = 0; i < neurons.size(); i++) {
            weights.add(neurons.get(i).getWeight(i));
        }
        return weights;
    }
    
    public List<Double> getGradients() {
        List<Double> gradients = new ArrayList<>();
        for (int i = 0; i < neurons.size(); i++) {
            gradients.add(neurons.get(i).getGradient());
        }
        return gradients;
    }
}
