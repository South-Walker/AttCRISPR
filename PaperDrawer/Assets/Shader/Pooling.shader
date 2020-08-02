Shader "Unlit/Pooling"
{
    Properties
    {
		_AvgColor("AvgPooling Color", Color) = (0, 0, 0, 1)
		_MaxColor("MaxPooling Color",  Color) = (0, 0, 0, 1)
    }
    SubShader
    {
        Pass
        {
		    ZTest Off
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            struct a2v
            {
                float4 vertex : POSITION;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
				float4 objectpos : TEXCOORD1;
            };


            v2f vert (a2v v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
				o.objectpos = v.vertex;
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                return 1;
            }
            ENDCG
        }
    }
}
