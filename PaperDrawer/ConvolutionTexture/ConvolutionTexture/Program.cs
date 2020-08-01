using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Threading.Tasks;
using System.Drawing;

namespace ConvolutionTexture
{
    class Program
    {
        static StringBuilder rootPath { get { return new StringBuilder(@"../../../../DrawerData/"); } }
        static string sgRNADirectory = rootPath.Append("Convolution/").ToString();
        static string outputPath = rootPath.Append("convolution{0}.png").ToString(); 
        static void Main(string[] args)
        {
            DirectoryInfo dirinfo = new DirectoryInfo(sgRNADirectory);
            int imgcount = 0;
            foreach (var fileinfo in dirinfo.GetFiles())
            {
                StreamReader sr = fileinfo.OpenText();
                var ss = sr.ReadToEnd().Split(',');
                int ssindex = 0;
                double[,] convoutput = new double[21, 4];
                double max = double.MinValue;
                double min = double.MaxValue;
                for (int i = 0; i < convoutput.GetLength(0); i++)
                {
                    for (int j = 0; j < convoutput.GetLength(1); j++)
                    {
                        convoutput[i, j] = Convert.ToDouble(ss[ssindex++]);
                        max = (max > convoutput[i, j]) ? max : convoutput[i, j];
                        min = (min > convoutput[i, j]) ? convoutput[i, j] : min;
                    }
                }
                double scale = 1.0 / (max - min);
                for (int i = 0; i < convoutput.GetLength(0); i++)
                {
                    for (int j = 0; j < convoutput.GetLength(1); j++)
                    {
                        convoutput[i, j] += min;
                        convoutput[i, j] *= scale;
                    }
                }
                Bitmap img = new Bitmap(1024, 1024);
                for (int w = 0; w < img.Width; w++)
                {
                    for (int h = 0; h < img.Height; h++)
                    {
                        int i = w * convoutput.GetLength(0) / (img.Width + 1);
                        int j = h * convoutput.GetLength(1) / (img.Height + 1);
                        int grey = (int)(convoutput[i, j] * 255);
                        img.SetPixel(h, w, Color.FromArgb(grey, grey, grey));
                    }
                }
                img.Save(string.Format(outputPath, imgcount));
                imgcount++;
            }
        }
    }
}
