using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
[ExecuteInEditMode]
public class BlendCamera : MonoBehaviour
{
    public RenderTexture rt_posteff;
    public Shader blend;
    private Material m_blend;
    // Start is called before the first frame update
    void Start()
    {
        
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
        
    }
}
