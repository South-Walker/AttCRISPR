using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
[ExecuteInEditMode]
[RequireComponent(typeof(Camera))]
public class BlendCamera : MonoBehaviour
{
    public RenderTexture rt_posteff;
    public Camera posteffcam;
    public Shader blend;
    private Material m_blend;

    private int camwidth, camheight;
    private Camera cam;
    // Start is called before the first frame update
    void Awake()
    {
        cam = GetComponent<Camera>();
        camwidth = cam.pixelWidth;
        camheight = cam.pixelHeight;
        rt_posteff = new RenderTexture(camwidth, camheight, 0);
        posteffcam.targetTexture = rt_posteff;
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
        if (cam.pixelHeight != camheight || cam.pixelWidth != camwidth)
        {
            camwidth = cam.pixelWidth;
            camheight = cam.pixelHeight;
            rt_posteff = new RenderTexture(camwidth, camheight, 0);
            posteffcam.targetTexture = rt_posteff;
        }
    }
}
