Shader "Unlit/BlendOP"
{
	Properties
	{
		_PosteffTex("Tex_Posteff", 2D) = "white" {}
		_SourceTex("Tex_Source",2D) = "white" {}
	}
		SubShader
	{
		Pass
		{
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag


			#include "UnityCG.cginc"

			struct appdata
			{
				float4 vertex : POSITION;
				float2 uv : TEXCOORD0;
			};

			struct v2f
			{
				float2 posteffuv : TEXCOORD0;
				float2 sourceuv : TEXCOORD1;
				float4 vertex : SV_POSITION;
			};

			sampler2D _PosteffTex;
			float4 _PosteffTex_ST;
			sampler2D _SourceTex;
			float4 _SourceTex_ST;

			v2f vert(appdata v)
			{
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				o.sourceuv = TRANSFORM_TEX(v.uv, _SourceTex);
				o.posteffuv = TRANSFORM_TEX(v.uv, _PosteffTex);
				return o;
			}

			fixed4 frag(v2f i) : SV_Target
			{
				fixed4 poseff = tex2D(_PosteffTex,i.posteffuv);
				fixed4 source = tex2D(_SourceTex, i.sourceuv);
				fixed4 col = lerp(poseff, source, source.a);
				return col;
			}
			ENDCG
		}
	}
}
