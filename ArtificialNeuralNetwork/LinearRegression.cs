using System;
using System.Linq;

namespace ArtificialNeuralNetwork
{
    public class LinearRegression
    {
        public double CoefficientDetermination { get; private set; }
        private double _valueM;
        private double _valueB;
        private double _error;

        public LinearRegression()
        {
            _valueM = 0d;
            _valueB = 0d;
            _error = 0.1d;
    }

        public double Run(double input)
        {
            return (_valueM * input) + _valueB;
        }

        public void Training(double[] inputs, double[] outputs)
        {
            double meanValueX = MeanValue(inputs);
            double meanValueY = MeanValue(outputs);
            double meanSquareValueX = MeanSquareValue(inputs);
            double meanValueXY = MeanValueArrays(inputs, outputs);

            _valueM = (meanValueXY - (meanValueX * meanValueY)) / (meanSquareValueX - (Math.Pow(meanValueX, 2)));
            _valueB = meanValueY - (_valueM * meanValueX);

            CoefficientDetermination = GetCoefficientDetermination(inputs, outputs, meanValueY);
        }

        private double MeanValue(double[] input)
        {
            return input.Aggregate((a, b) => a + b) / input.Length;
        }

        private double MeanSquareValue(double[] input)
        {
            double[] inputSquare = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
                inputSquare[i] = Math.Pow(input[i], 2);
            return inputSquare.Aggregate((a, b) => a + b) / inputSquare.Length;
        }

        private double MeanValueArrays(double[] inputX, double[] outputY)
        {
            var joinSumValues = inputX.Select((x, index) => x * outputY[index]).ToArray();
            return joinSumValues.Aggregate((a, b) => a + b) / joinSumValues.Length;
        }

        private double GetCoefficientDetermination(double[] inputs, double[] outputs, double meanValueY)
        {
            double ssLine = 0d;
            double ssOutput = 0d;
            for (int i = 0; i < inputs.Length; i++)
            {
                ssLine += Math.Pow((outputs[i] - (_valueM * inputs[i] + _valueB)), 2);
                ssOutput += Math.Pow((outputs[i] - meanValueY), 2);
            }

            return 1 - (ssLine / ssOutput);
        }
    }
}
