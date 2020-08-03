using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[ExecuteInEditMode]
[RequireComponent(typeof(Camera))]
public class BlendOP : MonoBehaviour
{
    public RenderTexture rt_posteff;
    public Camera posteffCam;
    public Shader blend;
    private Vector4 rt_posteff_st;
    [Range(-1, 1)]
    public float offset_x;
    [Range(-1, 1)]
    public float offset_y;
    private Material m_blend;

    private int camwidth, camheight;
    private Camera mainCam;
    // Start is called before the first frame update
    void Awake()
    {
        mainCam = GetComponent<Camera>();
        camwidth = mainCam.pixelWidth;
        camheight = mainCam.pixelHeight;
        rt_posteff = new RenderTexture(camwidth, camheight,0);
        posteffCam.targetTexture = rt_posteff;
    }
    private void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        if (m_blend == null)
        {
            m_blend = new Material(blend);
        }
        rt_posteff_st.z = offset_x;
        rt_posteff_st.w = offset_y;
        rt_posteff_st.x = 1;// - offset_x;
        rt_posteff_st.y = 1;// - offset_y;
        m_blend.SetTexture("_PosteffTex", rt_posteff);
        m_blend.SetVector("_PosteffTex_ST", rt_posteff_st);
        m_blend.SetTexture("_SourceTex", source);
        Graphics.Blit(source, destination, m_blend);
    }
    // Update is called once per frame
    void Update()
    {
        if (mainCam.pixelHeight != camheight || mainCam.pixelWidth != camwidth)
        {
            camwidth = mainCam.pixelWidth;
            camheight = mainCam.pixelHeight;
            rt_posteff = new RenderTexture(camwidth, camheight, 0);
            posteffCam.targetTexture = rt_posteff;
        }
    }
}
