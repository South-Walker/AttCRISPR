using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;

namespace EmbeddingTexture
{
    struct HSI
    {
        public double H, S, I;
        private double agreeH { get { return 180 * H / Math.PI; } }
        public Color toRGBA()
        {
            double r = 0,
                g = 0,
                b = 0;
            if (agreeH < 120)
            {
                b = I * (1 - S);
                r = I * (1 +
                    (
                    S * Math.Cos(H) / Math.Cos(60.0 / 180.0 * Math.PI - H)
                    ));
                g = 3 * I - (r + b);
            }
            else if (agreeH < 240)
            {
                double newH = H - 120.0 / 180.0 * Math.PI;
                r = I * (1 - S);
                g = I * (1 +
                    (
                    S * Math.Cos(newH) / Math.Cos(60.0 / 180.0 * Math.PI - newH)
                    ));
                b = 3 * I - (r + g);
            }
            else
            {
                double newH = H - 240.0 / 180.0 * Math.PI;
                g = I * (1 - S);
                b = I * (1 +
                    (
                    S * Math.Cos(newH) / Math.Cos(60.0 / 180.0 * Math.PI - newH)
                    ));
                r = 3 * I - (g + b);
            }
            return Color.FromArgb((int)(r * 256), (int)(g * 256), (int)(b * 256));
        }
        static public HSI fromRGBA(Color color)
        {
            return fromRGBA(color.R, color.G, color.B);
        }
        static public HSI fromRGBA(int R, int G, int B)
        {
            double r = (double)R / 256.0;
            double g = (double)G / 256.0;
            double b = (double)B / 256.0;
            HSI hsi = new HSI();
            double theta = Math.Acos(
                ((r-g)+(r-b))/2.0
                / Math.Sqrt((r-g)*(r-g)+(r-b)*(g-b))
                );
            hsi.H = (b <= g) ? theta : 2 * Math.PI - theta;
            hsi.I = (double)(r + g + b) / 3;
            double minrgb = Math.Min(r, Math.Min(g, b));
            hsi.S = 1 -
                3 / (r + g + b) * minrgb;
            return hsi;
        }
    }
    class Program
    {
        static StringBuilder rootpath { get { return new StringBuilder(@"../../../../../PSD/"); } }
        static string circlepath = rootpath.Append(@"Circle.png").ToString();
        static Bitmap circle = (Bitmap)Bitmap.FromFile(circlepath);
        static string deepsoildcolsavepath = rootpath.Append(@"deepfilled.png").ToString();
        static string soildcolsavepath = rootpath.Append(@"filled.png").ToString();
        static string withlinesavepath1 = rootpath.Append(@"withline1.png").ToString();
        static string withlinesavepath2 = rootpath.Append(@"withline2.png").ToString();
        static string mergepath = rootpath.Append(@"{0}merge.png").ToString();
        static string nowworkingmergepath;
        static Dictionary<char, Color> base2color = new Dictionary<char, Color>
        {
            {'A',Color.Red },
            {'G',Color.Blue },
            {'C',Color.Green },
            {'T',Color.FromArgb(255,255,128,0) }
        };
        static void Main(string[] args)
        {
            foreach (var item in base2color) 
            {
                nowworkingmergepath = string.Format(mergepath, item.Key);
                Pipeline(item.Value);
            }
            Console.WriteLine(Directory.GetCurrentDirectory());
        }
        static void Pipeline(Color color)
        {
            HSI hsi = HSI.fromRGBA(color);
            hsi.S *= 0.5;
            Color deepcolor = hsi.toRGBA();
            //Bitmap filledcircle = FillCircle(color);
            //filledcircle.Save(soildcolsavepath);
            
            Bitmap deepfilledcircle = FillCircle(deepcolor);
            //deepfilledcircle.Save(deepsoildcolsavepath);
            Bitmap fillwithline1 = FillCircleWithLine(color,true);
            //fillwithline1.Save(withlinesavepath1);
            Bitmap fillwithline2 = FillCircleWithLine(color, false);
            //fillwithline2.Save(withlinesavepath2);
            Bitmap[] imgs = new Bitmap[] { deepfilledcircle, circle, fillwithline1, fillwithline2 };
            Bitmap merget = Merget(imgs);
            merget.Save(nowworkingmergepath);
        }
        static Bitmap Merget(Bitmap[] imgs)
        {
            double exwidth = 1;
            double exheight = 0.2;
            double distance = exwidth / (imgs.Length + 1);
            int circlewidth = imgs[0].Width;
            int circleheight = imgs[0].Height;
            Bitmap merget = new Bitmap((int)(imgs.Length + exwidth) * circlewidth,
                (int)(circleheight * (1 + exheight)));
            for (int x = 0; x < merget.Width; x++)
            {
                for (int y = 0; y < merget.Height; y++)
                {
                    merget.SetPixel(x, y, Color.White);
                }
            }
            int xbeginat = (int)(circlewidth * distance);
            int ybeginat = (int)(circleheight * exheight / 2);
            for (int i = 0; i < imgs.Length; i++)
            {
                for (int x = 0; x < imgs[i].Width; x++)
                {
                    for (int y = 0; y < imgs[i].Height; y++)
                    {
                        merget.SetPixel(xbeginat + x, ybeginat + y,
                            imgs[i].GetPixel(x, y));
                    }
                }
                xbeginat += circlewidth + (int)(circlewidth * distance);
            }
            return merget;
        }
        static bool isinline(int x, int y, int distancestep, int distancelimit, int stepnum)
        {
            int dxy = Math.Abs(x - y);
            for (int i = 0; i < (stepnum+1)/2; i++)
            {
                int mindis, maxdis;
                if (stepnum % 2 == 0)
                {
                    mindis = (i + 1) * distancestep + i * distancelimit;
                    maxdis = mindis + distancelimit;
                }
                else
                {
                    mindis = i * (distancelimit + distancestep) - distancelimit / 2;
                    maxdis = distancelimit + mindis;
                }
                if (dxy >= mindis && dxy <= maxdis)
                    return true;
            }
            return false;
        }
        static Bitmap FillCircleWithLine(Color color, bool isbottomtotop = true)
        {
            int distancestep = 100;
            int distancelimit = 20;
            //偶数好像算错呢
            int stepnum = 5;
            int limit = 50;
            int circlewidth = circle.Width;
            int circleheight = circle.Height;
            Bitmap filledwithlinecircle = new Bitmap(circlewidth, circleheight);
            bool isincircle = false;
            bool isinedge = false;
            for (int w = 0; w < circlewidth; w++)
            {
                isincircle = false;
                isinedge = false;

                for (int h = 0; h < circleheight; h++)
                {
                    Color now = circle.GetPixel(w, h);

                    if ((now.R + now.G + now.B) < limit)
                    {
                        isinedge = true;
                    }
                    else
                    {
                        if (isinedge)
                        {
                            isincircle = !isincircle;
                        }
                        isinedge = false;
                    }
                    if (isincircle && !isinedge)
                    {
                        if (
                            (isbottomtotop) ?
                            isinline(w, h, distancestep, distancelimit, stepnum) :
                            isinline(w, circleheight - h, distancestep, distancelimit, stepnum))  
                        {
                            filledwithlinecircle.SetPixel(w, h, Color.Black);
                        }
                        else
                        {
                            filledwithlinecircle.SetPixel(w, h, color);
                        }
                    }
                    else
                    {
                        filledwithlinecircle.SetPixel(w, h, now);
                    }
                }
                //相切
                if (isincircle)
                {
                    int back = circleheight - 1;
                    Color now;
                    do
                    {
                        now = circle.GetPixel(w, back);
                        filledwithlinecircle.SetPixel(w, back, now);
                        back--;
                    } while ((now.R + now.G + now.B) > limit);
                }
            }
            return filledwithlinecircle;
        }
        static Bitmap FillCircle(Color color)
        {
            int limit = 50;
            int circlewidth = circle.Width;
            int circleheight = circle.Height;
            Bitmap filledcircle = new Bitmap(circlewidth, circleheight);
            bool isincircle = false;
            bool isinedge = false;
            for (int w = 0; w < circlewidth; w++)
            {
                isincircle = false;
                isinedge = false;

                for (int h = 0; h < circleheight; h++)
                {
                    Color now = circle.GetPixel(w,h);

                    if ((now.R + now.G + now.B) < limit)
                    {
                        isinedge = true;
                    }
                    else
                    {
                        if (isinedge)
                        {
                            isincircle = !isincircle;
                        }
                        isinedge = false;
                    }
                    if (isincircle&&!isinedge)
                    {
                        filledcircle.SetPixel(w, h, color);
                    }
                    else
                    {
                        filledcircle.SetPixel(w, h, now);
                    }
                }
                //相切
                if (isincircle)
                {
                    int back = circleheight - 1;
                    Color now;
                    do
                    {
                        now = circle.GetPixel(w, back);
                        filledcircle.SetPixel(w, back, now);
                        back--;
                    } while ((now.R + now.G + now.B) > limit);
                }
            }
            return filledcircle;
        }
    }
}
