using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ArtificialNeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            #region Linear Regression
            Console.WriteLine("Test with Linear Regression");
            double[] input = new double[] { -2, -1, 1, 4 };
            double[] output = new double[] { -3, -1, 2, 3 };
            LinearRegression linearRegression = new LinearRegression();
            linearRegression.Training(input, output);
            Console.WriteLine("Test: " + linearRegression.Run(0.5d));
            Console.WriteLine("Coefficient Determination: " + linearRegression.CoefficientDetermination);
            Console.WriteLine("------------------------------");
            #endregion

            #region Perceptron

            #region AND Gate
            double[,] inputAnd = new double[,] { { 1, 0 }, { 1, 1 }, { 0, 1 }, { 0, 0 } };
            int[] outputAnd = new int[] { 0, 1, 0, 0 };

            Perceptron p1 = new Perceptron();
            p1.Training(inputAnd, outputAnd);
            Console.WriteLine("Test with Perceptron");
            Console.WriteLine("AND Gate");
            Console.WriteLine("Iteration of training: " + p1.Iteration);
            Console.WriteLine("Test 1: " + p1.Run(new double[,] { { 1, 0 } }));
            Console.WriteLine("Test 2: " + p1.Run(new double[,] { { 1, 1 } }));
            #endregion

            #region OR Gate
            double[,] inputOr = new double[,] { { 1, 0 }, { 1, 1 }, { 0, 1 }, { 0, 0 } };
            int[] outputOr = new int[] { 1, 1, 1, 0 };

            Perceptron p2 = new Perceptron();
            p2.Training(inputOr, outputOr);
            Console.WriteLine("OR Gate");
            Console.WriteLine("Iteration of training: " + p2.Iteration);
            Console.WriteLine("Test 1: " + p2.Run(new double[,] { { 0, 1 } }));
            Console.WriteLine("Test 2: " + p2.Run(new double[,] { { 0, 0 } }));
            #endregion

            #region NOT Gate
            double[,] inputNot = new double[,] { { 1 }, { 0 } };
            int[] outputNot = new int[] { 0, 1 };

            Perceptron p3 = new Perceptron(1);
            p3.Training(inputNot, outputNot);
            Console.WriteLine("NOT Gate");
            Console.WriteLine("Iteration of training: " + p3.Iteration);
            Console.WriteLine("Test 1: " + p3.Run(new double[,] { { 0 } }));
            Console.WriteLine("Test 2: " + p3.Run(new double[,] { { 1 } }));
            #endregion

            #endregion

            #region Multilayer Parceptron
            Console.WriteLine("----------------------");
            Console.WriteLine("Test with MLP");
            MultilayerPerceptron mlp = new MultilayerPerceptron(2, 6, 1);
            mlp.Training(new double[,] { { 1, 1 }, { 1, 0 }, { 0, 0 }, { 0, 1 } }, new double[] {1, 1, 0, 1});
            Console.WriteLine("AND Gate: " + mlp.Run(new double[] { 1, 1 }).FirstOrDefault());
            #endregion

            Console.ReadKey();
        }
    }
}
