using System;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Complex32;

namespace Lab4
{
	class Program
	{
		static void Main(string[] args)
		{
			//float[,] X = {
			//	{ 22.4f, 17.1f,22f},
			//	{224.2f,17.1f,23f },
			//	{151.8f, 14.9f,21.5f },
			//	{147.3f,13.6f, 28.7f},
			//	{152.3f,10.5f,10.2f}
			//};
			//float[,] Y = {
			//		{46.8f,4.4f,11.1f},
			//		{29f,5.5f,6.1f },
			//		{52.1f,4.2f,11.8f},
			//		{37.1f,5.5f,11.9f},
			//		{64f,4.2f,12.9f}
			//}; //Examples
			//float[,] Z1 = { { 75f, 9.6f, 18.5f } };//Examples
			//float[,] Z2 = { { 95f, 12.5f, 16.1f } };

			float[,] X =
			{
				{ 0.2f, 3.31f ,25.5f,85f},
				{0.15f,4.52f,32.3f,85f },
				{0.18f,4.43f,31.2f,95f },
				{0.17f, 3.73f,27.7f,88f},
				{ 0.22f,3.94f,26.8f,90f },
				{0.16f,3.83f,32.5f,85f },
				{0.17f,4.54f,22.2f,87f }
			};
			float[,] Y =
			{
				{0.25f,3.25f,25.3f,92f },
				{0.48f,3.18f,24.0f,93f },
				{0.55f,3.45f,19.8f,98f },
				{ 0.18f,4.18f, 26.1f,102},
				{0.35f,3.35f,19.0f,109 },
				{0.43f,3.43f,18.2f,91f },
				{0.29f,4.29f,24.3f,94f }
			};
			float[,] Z1 = { { 0.3f, 4.23f, 28.2f, 100f } };//10 var;
			float[,] Z2 = { { 0.2f, 4.02f, 25.3f, 99f } };

			float[] Xmid = GetMatrixMid(X);
			float[] Ymid = GetMatrixMid(Y);

			Console.WriteLine("X: ");
			PrintMatrix(X);
			Console.WriteLine("X mid values: ");
			PrintMatrix(Xmid);
			Console.WriteLine("Y: ");
			PrintMatrix(Y);
			Console.WriteLine("Y mid values: ");
			PrintMatrix(Ymid);
			Console.WriteLine("Z1: ");
			PrintMatrix(Z1);
			Console.WriteLine("Z2: ");
			PrintMatrix(Z2);

			Console.WriteLine("Skj(x)");
			float[,] CovMatrixX = GetCovMatrix(X, Xmid);
			Console.WriteLine("Skj(y)");
			float[,] CovMatrixY = GetCovMatrix(Y, Ymid);
			Console.WriteLine("X covariative matrix: ");
			PrintMatrix(CovMatrixX);
			Console.WriteLine("Y covariative matrix: ");
			PrintMatrix(CovMatrixY);

			float[,] multipliedXMatrix = MultiplyMatrixByNumber(CovMatrixX, X.GetLength(0));
			float[,] multipliedYMatrix = MultiplyMatrixByNumber(CovMatrixY, Y.GetLength(0));
			Console.WriteLine("X matrix multiplied by n: ");
			PrintMatrix(multipliedXMatrix);
			Console.WriteLine("Y matrix multiplied by n: ");
			PrintMatrix(multipliedYMatrix);

			float[,] matrixSum = SumTwoMatrix(multipliedXMatrix, multipliedYMatrix);
			Console.WriteLine("Sum of two matrix: ");
			PrintMatrix(matrixSum);

			float[,] unmovedMarkOfUnatedMatrix = MultiplyMatrixByNumber(matrixSum, 1f/(X.GetLength(0) + Y.GetLength(0) - 2f));
			Console.WriteLine("Unmoved mark of united matrix: ");
			PrintMatrix(unmovedMarkOfUnatedMatrix);

			float[,] inverseMatrix = GetInvertedMatrix(unmovedMarkOfUnatedMatrix);
			Console.WriteLine("Inversed Matrix: ");
			PrintMatrix(inverseMatrix);

			float[] midVectorofCoefficients = GetMidVector(Xmid, Ymid);
			Console.WriteLine("Midle vector of coeficients:");
			PrintMatrix(midVectorofCoefficients);
			float[] markVector = GetMarkVector(inverseMatrix, midVectorofCoefficients);
			Console.WriteLine("Vector of marks coefitients of discriminative function: ");
			PrintMatrix(markVector);

			float[] Xmarks =GetMarkVectorOfDiscriminativeFunc(X, markVector);
			float[] Ymarks = GetMarkVectorOfDiscriminativeFunc(Y, markVector);
			Console.WriteLine("Marked vectors of discriminative functions Uxi:");
			PrintMatrix(Xmarks);
			Console.WriteLine("Marked vectors of discriminative functions Uyi:");
			PrintMatrix(Ymarks);

			Console.WriteLine("Middle X marks: ");
			Console.WriteLine(GetMidleMark(Xmarks));
			Console.WriteLine("Middle Y marks: ");
			Console.WriteLine(GetMidleMark(Ymarks));

			float disConst = (GetMidleMark(Xmarks) + GetMidleMark(Ymarks)) / 2f;
			Console.WriteLine("Discriminative const: ");
			Console.WriteLine(disConst);

			Console.WriteLine("Mark of Z1 function: ");
			Console.WriteLine(GetMarkOfDisFunc(Z1, markVector));
			Console.WriteLine("Mark of Z2 function: ");
			Console.WriteLine(GetMarkOfDisFunc(Z2, markVector));

			if(GetMarkOfDisFunc(Z1, markVector) < disConst)
			{
				Console.WriteLine("Z1 is belong to second sample");
			}
			else
			{
				Console.WriteLine("Z1 is belong to first sample");
			}

			if(GetMarkOfDisFunc(Z2, markVector) < disConst)
			{
				Console.WriteLine("Z2 is belong to second sample");
			}
			else
			{
				Console.WriteLine("Z2 is belong to first sample");
			}
		}

		public static float GetMarkOfDisFunc(float[,] Z, float[] midleMarks)
		{
			float res = 0f;

			for(int i =0; i < Z.Length; i++)
			{
				res += Z[0,i] * midleMarks[i];
			}
			return res;
		}

		public static float GetMidleMark(float[] marks)
		{
			float res = 0f;
			for(int i =0; i < marks.Length; i++)
			{
				res += marks[i];
			}
			return res/marks.Length;
		}

		public static float[] GetMarkVectorOfDiscriminativeFunc(float[,] func, float[] markVector)
		{
			Matrix matrix = new DenseMatrix(func.GetLength(0), func.GetLength(1));
			for (int i = 0; i < matrix.RowCount; i++)
			{
				for (int j = 0; j < matrix.ColumnCount; j++)
				{
					matrix[i, j] = func[i, j];
				}
			}

			Vector vector = new DenseVector(markVector.Length);
			for (int i = 0; i < markVector.Length; i++)
			{
				vector[i] = markVector[i];
			}

			Vector resVector = (Vector)matrix.Multiply(vector);
			float[] res = new float[resVector.Count];
			for (int i = 0; i < res.Length; i++)
			{
				res[i] = resVector[i].Real;
			}
			return res;
		}

		public static float[] GetMarkVector(float[,] inverseMatrix, float[] midVector)
		{
			float[] res = midVector;

			Matrix matrix = new DenseMatrix(inverseMatrix.GetLength(0), inverseMatrix.GetLength(1));
			for (int i = 0; i < matrix.RowCount; i++)
			{
				for (int j = 0; j < matrix.ColumnCount; j++)
				{
					matrix[i, j] = inverseMatrix[i, j];
				}
			}

			Vector vector = new DenseVector(midVector.Length);
			for(int i = 0; i < midVector.Length; i++)
			{
				vector[i] = midVector[i];
			}

			Vector resVector = (Vector)matrix.Multiply(vector);

			for (int i = 0; i < midVector.Length; i++)
			{
				res[i] = resVector[i].Real;
			}

			return res;
		}

		public static float[] GetMidVector(float[] midX, float[] midY)
		{
			float[] res = midX;
			for(int i = 0; i < midX.Length; i++)
			{
				res[i] = midX[i] - midY[i];
			}
			return res;
		}

		public static float[,] GetInvertedMatrix(float[,] unmovedMarkOfUnatedMatrix)
		{
			Matrix matrix = new DenseMatrix(unmovedMarkOfUnatedMatrix.GetLength(0), unmovedMarkOfUnatedMatrix.GetLength(1));

			for (int i = 0; i < matrix.RowCount; i++)
			{
				for (int j = 0; j < matrix.ColumnCount; j++)
				{
					matrix[i, j] = unmovedMarkOfUnatedMatrix[i, j];
				}
			}
			matrix = (Matrix)matrix.Inverse();
			float[,] inverseMatrix = unmovedMarkOfUnatedMatrix;
			for (int i = 0; i < inverseMatrix.GetLength(0); i++)
			{
				for (int j = 0; j < inverseMatrix.GetLength(1); j++)
				{
					inverseMatrix[i, j] = matrix[i, j].Real;
				}
			}
			return inverseMatrix;
		}

		private static float[,] SumTwoMatrix(float[,] matrix1, float[,] matrix2)
		{
			float[,] res = matrix1;
			for (int i = 0; i < matrix1.GetLength(0); i++)
			{
				for (int j = 0; j < matrix1.GetLength(1); j++)
				{
					res[i, j] += matrix2[i, j]; 
				}
			}
			return res;
		}

		public static float[,] MultiplyMatrixByNumber(float[,] matrix, float number)
		{
			float[,] multipliedMatrix = matrix;
			for(int i = 0; i< matrix.GetLength(0); i++)
			{
				for(int j = 0; j< matrix.GetLength(1); j++)
				{
					multipliedMatrix[i, j] = matrix[i, j] * number;
				}
			}
			return multipliedMatrix;
		}

		public static float[] GetMatrixMid(float[,] matrix)
		{
			float[] midValues = new float[matrix.GetLength(1)];
			for(int i = 0; i < matrix.GetLength(1); i++)
			{
				float sum = 0f;
				for(int j = 0; j < matrix.GetLength(0); j++)
				{
					sum += matrix[j, i];
				}
				midValues[i] = sum/matrix.GetLength(0);
			}
			Console.WriteLine();
			return midValues;
		}

		public static float[,] GetCovMatrix(float[,] matrix, float[] matrixMid)
		{
			float[,] covMatrix = new float[matrix.GetLength(0) + 1,(matrix.GetLength(1) * (matrix.GetLength(1) + 1)) / 2];
			int i = 0;
			int j = 0;
			int f = 0;
			for (int k = 0; k < covMatrix.GetLength(0)-1; k++)
			{
				for (int n = 0; n < covMatrix.GetLength(1); n++)
				{
					float matrixNI = matrix[k, i];
					float matrixMidI = matrixMid[i];
					float matrixNJ = matrix[k, j];
					float matrixMidJ = matrixMid[j];
					covMatrix[k, n] = (matrixNI - matrixMidI) * (matrixNJ - matrixMidJ);
					j++;
					if (j == matrix.GetLength(1))
					{
						i++;
						j = i;
					}
				}
				i = 0;
				j = 0;
			}

			for (int k = 0; k < covMatrix.GetLength(1); k++)
			{
				float sum = 0f;
				for (int n = 0; n < covMatrix.GetLength(0); n++)
				{
					sum += covMatrix[n, k];
				}
				int count = covMatrix.GetLength(0)-1;
				covMatrix[covMatrix.GetLength(0) - 1, k] = sum / count;
			}
			PrintMatrix(covMatrix);

			float[,] res = new float[matrix.GetLength(1), matrix.GetLength(1)];

			int m = 0;
			for(int k = 0; k< res.GetLength(1); k++)
			{
				for(int n = k; n < res.GetLength(1); n++)
				{
					res[k, n] = covMatrix[covMatrix.GetLength(0) - 1, m];
					res[n, k] = covMatrix[covMatrix.GetLength(0) - 1, m];
					m++;
				}
			}
			return res;
		}

		public static void PrintMatrix(float[,] matrix)
		{
			int numRows = matrix.GetLength(0);
			int numCols = matrix.GetLength(1);

			for (int row = 0; row < numRows; row++)
			{
				for (int col = 0; col < numCols; col++)
				{
					Console.Write(" {0,12}", matrix[row, col]);
				}
				Console.WriteLine();
			}
			Console.WriteLine();
		}

		public static void PrintMatrix(float[] matrix)
		{
			for (int i = 0; i < matrix.Length; i++)
			{
				Console.Write(" {0,12}", matrix[i]);
			}
			Console.WriteLine("\n");
		}
	}
}
