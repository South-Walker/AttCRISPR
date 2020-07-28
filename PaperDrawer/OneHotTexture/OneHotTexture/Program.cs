using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;

namespace OneHotTexture
{
    class Program
    {
        static char[] sgRNA = "GTTGAGAAGGACCGCCACAAC".ToCharArray();
        static Dictionary<char, int[]> base2onehot = new Dictionary<char, int[]>
        {
            {'A',new int[4]{1,0,0,0} },
            {'T',new int[4]{0,1,0,0} },
            {'G',new int[4]{0,0,1,0} },
            {'C',new int[4]{0,0,0,1} }

        };
        static string outputpath = @"./onehot.png";
        static void Main(string[] args)
        {
            Bitmap img = new Bitmap(1024, 1024);
            int[,] onehot = new int[21, 4];
            for (int i = 0; i < onehot.GetLength(0); i++)
            {
                var nowonehot = base2onehot[sgRNA[i]];
                for (int j = 0; j < onehot.GetLength(1); j++)
                {
                    onehot[i, j] = nowonehot[j];
                }
            }
            for (int w = 0; w < img.Width; w++)
            {
                for (int h = 0; h < img.Height; h++)
                {
                    int i = w * onehot.GetLength(0) / (img.Width + 1);
                    int j = h * onehot.GetLength(1) / (img.Height + 1);
                    img.SetPixel(h, w, (onehot[i, j] == 0) ? Color.Black : Color.White);
                }
            }
            img.Save(outputpath);
        }
    }
}
