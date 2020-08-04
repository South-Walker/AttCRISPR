using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[ExecuteInEditMode]
public class ConnectRect : MonoBehaviour
{
    public GameObject cude1,cude2;
    public Camera cam1, cam2, maincam;
    public Vector2 offset1;
    public Material mat;
    private Vector2 cude1FrontTopRightinScreen;
    private Vector2 cude1FrontBottomRightinScreen;
    private Vector2 cude2FrontTopLeftScreen;
    private Vector2 cude2FrontBottomLeftScreen;

    // Start is called before the first frame update
    void Start()
    {
        cude1FrontTopRightinScreen = GetScreenPos(new Vector4(0.5f, 0.5f, 0.5f, 1), cude1, cam1, offset1);
        cude1FrontBottomRightinScreen = GetScreenPos(new Vector4(0.5f, -0.5f, 0.5f, 1), cude1, cam1, offset1);
        cude2FrontTopLeftScreen = GetScreenPos(new Vector4(-0.5f, 0.5f, 0.5f, 1), cude2, cam2);
        cude2FrontBottomLeftScreen = GetScreenPos(new Vector4(-0.5f, -0.5f, 0.5f, 1), cude2, cam2);
    }
    private void OnGUI()
    {
    }
    private void OnPostRender()
    {
        GL.PushMatrix();
        GL.LoadOrtho();
        mat.SetPass(0);
        GL.Begin(GL.LINES);

        GL.Vertex3(cude1FrontTopRightinScreen.x / maincam.pixelWidth, 1-cude1FrontTopRightinScreen.y / maincam.pixelHeight, 0);
        GL.Vertex3(cude2FrontTopLeftScreen.x / maincam.pixelWidth, 1-0.5f-cude2FrontTopLeftScreen.y / maincam.pixelHeight, 0);

        GL.End();
        GL.PopMatrix();
    }
    Vector2 GetScreenPos(Vector4 objectpos, GameObject game, Camera cam)
    {
        return GetScreenPos(objectpos, game, cam, Vector2.zero);
    }
    Vector2 GetScreenPos(Vector4 objectpos, GameObject game, Camera cam, Vector2 offset)
    {
        Vector3 worldPos = game.transform.localToWorldMatrix * objectpos;
        Vector2 camscreenPos = cam.WorldToScreenPoint(worldPos);
        Vector2 mainscreenPos = new Vector2(camscreenPos.x - offset.x * cam.pixelWidth,
            cam.pixelHeight - camscreenPos.y + offset.y * cam.pixelHeight);
        return mainscreenPos;
    }
}
