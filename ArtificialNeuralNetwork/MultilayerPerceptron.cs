using System;
using System.Collections.Generic;
using System.Linq;

namespace ArtificialNeuralNetwork
{
    public class MultilayerPerceptron
    {
        public int CodeHospital { get; set; }
        public List<Layer> Layers { get; set; }
        public List<double> Inputs { get; set; }
        public List<double> Outputs { get; set; }
        private double _input;
        private double _error = 0.005d;

        public MultilayerPerceptron(int input, int hidden, int output)
        {
            CodeHospital = 4;
            _input = input;
            Layers = new List<Layer>();
            Inputs = new List<double>();
            Outputs = new List<double>();

            // hidden layer
            Layers.Add(new Layer(input, hidden));
            // output layer
            Layers.Add(new Layer(hidden, output));
        }

        public double[] Run(double[] inputs)
        {
            foreach (Layer layer in Layers)
                inputs = CalculateLayer(layer, inputs);
            return inputs;
        }

        public void Training(double[,] inputs, double[] outputs)
        {
            long iteration = 0;
            double error;
            do
            {
                error = 0;
                int numberTrain = inputs.Length / inputs.GetLength(1);
                for (int i = 0; i < numberTrain; i++)
                {
                    double tagert = outputs[i];
                    double[] inputTest = new double[inputs.GetLength(1)];
                    for (int j = 0; j < inputs.GetLength(1); j++)
                        inputTest[j] = inputs[i, j];

                    double output = Run(inputTest).FirstOrDefault(k => k > 0);
                    Backpropagation(inputTest, output, tagert);

                    double delta = tagert - output;
                    error += Math.Pow(delta, 2);
                }

                iteration++;
                System.Diagnostics.Debug.WriteLine(error);
            } while (error >= _error);
        }

        private double[] CalculateLayer(Layer layer, double[] input)
        {
            double[] values = new double[layer.Neurons.Count];
            for (int i = 0; i < layer.Neurons.Count; i++)
            {
                var valueNeuron = layer.Neurons[i].LoadNeuron(new List<double>(input));
                values[i] = layer.Neurons[i].Activate(valueNeuron);
            }
            return values;
        }

        private void Backpropagation(double[] input, double output, double target)
        {
            // Go layers
            for (int i = (Layers.Count - 1); i >= 0; i--)
            {
                // Go neurons layer
                for (int j = 0; j < Layers[i].Neurons.Count; j++)
                {
                    List<double> outputsLeft = new List<double>();
                    outputsLeft.AddRange(i > 0 ? Layers[i - 1].Neurons.Select(t => t.Output) : input);

                    if (i == Layers.Count - 1)
                    {
                        Layers[i].Neurons[j].UpdateWeightsOutputLayer(output, outputsLeft, target);
                    }
                    else
                    {
                        Layers[i].Neurons[j].UpdateWeightsHiddenLayer(
                            Layers[i].Neurons[j].Output,
                            outputsLeft,
                            Layers[i + 1].Neurons.Select(k => k.Weight[j]).ToList(),
                            Layers[i + 1].Neurons.Select(k => k.Gradient).ToList());
                    }
                }
            }
        }
    }

    public class Layer
    {
        public List<Neuron> Neurons { get; private set; }
        public string Name { get; set; }

        public Layer(int inputs, int neurons)
        {
            Random random = new Random();
            Neurons = new List<Neuron>();
            for (int i = 0; i < neurons; i++)
                Neurons.Add(new Neuron(inputs, random));
        }
    }

    public class Neuron
    {
        public List<int> Code { get; set; }
        public string Name { get; set; }
        public double Output { get; set; }
        public List<double> Weight { get; private set; }
        public double Gradient { get; private set; }
        private double _euler = 2.718281828;
        private double _learningRate = 0.5;

        public Neuron(int inputs, Random rnd)
        {
            Name = Guid.NewGuid().ToString();
            Code = new List<int>();
            Weight = new List<double>();
            for (int i = 0; i < inputs; i++)
                Weight.Add(rnd.NextDouble() * 2 - 1);
        }

        public double Activate(double result)
        {
            Output = (1 / (1 + Math.Pow(_euler, -result)));
            return Output;
        }

        public double LoadNeuron(List<double> input)
        {
            if (input.Count == Weight.Count)
                return Weight.Select((weight, i) => input[i] * weight).Sum();
            else
                throw new Exception("Existe diferença entre os Inputs do neurônio e os pesos");
        }

        public void UpdateWeightsOutputLayer(double output, List<double> outputLeft, double target)
        {
            Gradient = output * ((1 - output) * (target - output));
            for (int i = 0; i < Weight.Count; i++)
                Weight[i] = Weight[i] + _learningRate * Gradient * outputLeft[i];
        }

        public void UpdateWeightsHiddenLayer(double output, List<double> outputLeft, List<double> weightRight, List<double> gradientRight)
        {
            Gradient = output * ((1 - output) * (weightRight.Select((weight, i) => gradientRight[i] * weight).Sum()));
            for (int i = 0; i < Weight.Count; i++)
                Weight[i] = Weight[i] + _learningRate * Gradient * outputLeft[i];

        }
    }
}
