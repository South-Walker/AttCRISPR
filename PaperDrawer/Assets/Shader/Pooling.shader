Shader "Unlit/Pooling"
{
	Properties
	{
		_Width("Width of Line", float) = 0
		_XScale("X Scale",float) = 1
		_YScale("Y Scale",float) = 1
		_ZScale("Z Scale",float) = 1
		_Color("Line Color",  Color) = (0, 0, 0, 1)
    }
    SubShader
    {
        Pass
		{
			ZTest Off ZWrite Off Cull Off
			Blend SrcAlpha OneMinusSrcAlpha
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			struct a2v
			{
				float4 vertex : POSITION;
			};

			struct v2f
			{
				float4 vertex : SV_POSITION;
				float4 objectpos : TEXCOORD1;
			};

			float _Width;
			float _XScale;
			float _YScale;
			float _ZScale;
			fixed4 _Color;
			v2f vert(a2v v)
			{
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				o.objectpos = v.vertex;
				return o;
			}

			fixed4 frag(v2f i) : SV_Target
			{
				fixed4 background = fixed4(0,0,0,0);
				fixed4 linecol = _Color;
				i.objectpos += 0.5;
				float xwidth = _XScale * _Width;
				float ywidth = _YScale * _Width;
				float zwidth = _ZScale * _Width;
				float isline = (i.objectpos.x < xwidth || 1 - i.objectpos.x < xwidth) ? 1 : 0;
				isline += (i.objectpos.y < ywidth || 1-i.objectpos.y < ywidth) ? 1 : 0;
				isline += (i.objectpos.z < zwidth || 1-i.objectpos.z < zwidth) ? 1 : 0;
				isline = (isline >= 2) ? 1 : 0;
                return lerp(background,linecol,isline);
            }
            ENDCG
        }
    }
}
