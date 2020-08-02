using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
[ExecuteInEditMode]
[RequireComponent(typeof(Camera))]
public class BlendCamera : MonoBehaviour
{
    public RenderTexture rt_posteff;
    public Camera posteffCam;
    public Shader blend;
    private Material m_blend;

    private int camwidth, camheight;
    private Camera mainCam;
    // Start is called before the first frame update
    void Awake()
    {
        mainCam = GetComponent<Camera>();
        camwidth = mainCam.pixelWidth;
        camheight = mainCam.pixelHeight;
        rt_posteff = new RenderTexture(camwidth, camheight, 0);
        posteffCam.targetTexture = rt_posteff;
    }
    private void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        if (m_blend == null)
        {
            m_blend = new Material(blend);
        }
        m_blend.SetTexture("_PosteffTex", rt_posteff);
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
