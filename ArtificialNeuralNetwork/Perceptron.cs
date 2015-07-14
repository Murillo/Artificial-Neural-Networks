using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ArtificialNeuralNetwork
{
    public class Perceptron
    {
        private double[] _weights;
        private double _learningRate;
        private double _error;
        private int _input;
        private bool _training;
        public int Iteration { get; set; }

        public Perceptron(int input = 2, double learning = 0.1f, double error = 0.1f, bool training = false)
        {
            _learningRate = learning;
            _error = error;
            _training = training;
            _input = input;

            LoadWeights();
        }

        private void LoadWeights()
        {
            Random r = new Random();
            _weights = new double[_input + 1];
            for (int i = 0; i < _input + 1; i++)
                _weights[i] = r.NextDouble();
        }

        public void Training(double[,] input, int[] output)
        {
            if (!_training)
            {
                while (_error != 0)
                {
                    _error = 0;
                    for (int i = 0; i < output.Count(); i++)
                    {
                        double[] ipt = new double[input.GetLength(1)];
                        for (int k = 0; k < input.GetLength(1); k++)
                            ipt[k] = input[i, k];

                        int outputTraining = CalculateOutput(ipt, _weights);

                        // Calculating error
                        double localError = output[i] - outputTraining;
                        if (localError != 0)
                        {
                            // Updating the weights
                            for (int k = 0; k < _weights.Count(); k++)
                            {
                                if (k == (_weights.Count() - 1))
                                    _weights[k] += _learningRate * localError * 1;
                                else
                                    _weights[k] += _learningRate * localError * input[i, k];
                            }
                        }
                        _error += Math.Abs(localError);
                    }
                    Iteration++;
                }
                _training = true;
            }
        }

        public int Run(double[,] input)
        {
            double[] ipt = new double[input.GetLength(1)];
            for (int k = 0; k < input.GetLength(1); k++)
                ipt[k] = input[0, k];

            return CalculateOutput(ipt, _weights);
        }

        private int CalculateOutput(double[] input, double[] weights)
        {
            double sum = 0f;
            for (int i = 0; i < weights.Length; i++)
            {
                if (i == (weights.Length - 1))
                    sum += 1 * weights[i];
                else
                    sum += input[i] * weights[i];
            }
            return (sum >= 0) ? 1 : 0;
        }
    }
}
