Shader "Unlit/Conv"
{
    Properties
    {
		_Intensity("Intensity",float) = 1
		_FrontTex("Front Texture",2D) = "white"{}
		_UpTex("Up Texture",2D) = "white"{}
		_RightTex("Right Texture",2D) = "white"{}
		_Base("Base",float) = 0
    }
    SubShader
    {
		Tags { "RenderType" = "Opaque" }
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
			#include "UnityCG.cginc"
            struct a2v
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
				float4 normal : NORMAL;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
				float4 normal : TEXCOORD1;
            };

			sampler2D _FrontTex;
			sampler2D _UpTex;
			sampler2D _RightTex;
			float _Intensity;
			float _Base;
            v2f vert (a2v v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
				o.uv = v.uv;
				o.normal = v.normal;
                return o;
            }

			fixed4 frag(v2f i) : SV_Target
			{
				i.normal = normalize(i.normal);
			    fixed4 up = tex2D(_UpTex, float2(i.uv.x,1-i.uv.y)) * -i.normal.z;
			    fixed4 right = tex2D(_RightTex, float2(1-i.uv.y,1-i.uv.x)) * i.normal.y;
			    fixed4 front = tex2D(_FrontTex, float2(i.uv.x,i.uv.y)) * i.normal.x;
				fixed4 col = up + right + front;
                return col * _Intensity + _Base;
            }
            ENDCG
        }
    }
}
