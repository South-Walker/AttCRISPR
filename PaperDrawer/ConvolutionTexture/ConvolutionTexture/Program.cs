using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Threading.Tasks;
using System.Drawing;
using System.Runtime.InteropServices;

namespace ConvolutionTexture
{
    class Program
    {
        static StringBuilder rootPath { get { return new StringBuilder(@"../../../../DrawerData/"); } }
        static string sgRNADirectory = rootPath.Append("Convolution/").ToString();
        static string allinput = rootPath.Append("Convolution/singleall.out").ToString();
        static string outputPath = rootPath.Append("convolution{0}.png").ToString();
        static void Main(string[] args)
        {
            double[,,] datas = new double[21, 4, 60];
            using (FileStream fs = new FileStream(allinput, FileMode.Open, FileAccess.Read))
            {
                StreamReader sr = new StreamReader(fs);
                var ss = sr.ReadToEnd().Split(',');
                int ssindex = 0;
                for (int i = 0; i < datas.GetLength(0); i++)
                {
                    for (int j = 0; j < datas.GetLength(1); j++)
                    {
                        for (int k = 0; k < datas.GetLength(2); k++)
                        {
                            datas[i, j, k] = Convert.ToDouble(ss[ssindex++]);  
                        }
                    }
                }
            }
            for (int k = 0; k < datas.GetLength(2); k++)
            {
                double max = double.MinValue;
                double min = double.MaxValue;
                for (int i = 0; i < datas.GetLength(0); i++)
                {
                    for (int j = 0; j < datas.GetLength(1); j++)
                    {
                        max = (max > datas[i, j, k]) ? max : datas[i, j, k];
                        min = (min > datas[i, j, k]) ? datas[i, j, k] : min;
                    }
                }
                double scale = (max - min);
                for (int i = 0; i < datas.GetLength(0); i++)
                {
                    for (int j = 0; j < datas.GetLength(1); j++)
                    {
                        datas[i, j, k] += min;
                        if (scale == 0)
                            datas[i, j, k] = 0.5;
                        else
                            datas[i, j, k] /= scale;
                    }
                }
            }
            CudeTexture(datas, 3, 20);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="datas"></param>
        /// <param name="outputdimnum"></param>
        /// 一个conv输出的维度
        /// <param name="inputdimnum"></param>
        static void CudeTexture(double[,,] datas, int outputdimnum, int inputdimnum)
        {
            int offsetk = 10;
            int totaldim = datas.GetLength(2);
            int cudecount = totaldim / inputdimnum;
            Bitmap img = new Bitmap(1024, 1024);
            for (int i = 0; i < cudecount; i++)
            {
                int kbegin = i * inputdimnum + offsetk;
                for (int w = 0; w < img.Width; w++)
                {
                    for (int h = 0; h < img.Height; h++)
                    {
                        int x = w * datas.GetLength(0) / (img.Width + 1);
                        int y = h * datas.GetLength(1) / (img.Height + 1);
                        int grey = (int)(datas[x, y, kbegin] * 255);
                        img.SetPixel(h, w, Color.FromArgb(grey, grey, grey));
                    }
                }
                img.Save(string.Format(outputPath,
                    string.Format("{0}_{1}", i, "front")
                    ));
                for (int w = 0; w < img.Width; w++)
                {
                    for (int h = 0; h < img.Height; h++)
                    {
                        int x = w * outputdimnum / (img.Width + 1);
                        int y = h * datas.GetLength(1) / (img.Height + 1);
                        int grey = (int)(datas[0, y, kbegin + x] * 255);
                        img.SetPixel(h, w, Color.FromArgb(grey, grey, grey));
                    }
                }
                img.Save(string.Format(outputPath,
                    string.Format("{0}_{1}", i, "right")
                    ));

                for (int w = 0; w < img.Width; w++)
                {
                    for (int h = 0; h < img.Height; h++)
                    {
                        int x = w * datas.GetLength(0) / (img.Width + 1);
                        int y = h * outputdimnum / (img.Height + 1);
                        int grey = (int)(datas[x, 0, kbegin + y] * 255);
                        img.SetPixel(h, w, Color.FromArgb(grey, grey, grey));
                    }
                }
                img.Save(string.Format(outputPath,
                    string.Format("{0}_{1}", i, "up")
                    ));
            }
        }
    }
}
