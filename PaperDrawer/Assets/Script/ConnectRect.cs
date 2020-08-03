using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(LineRenderer))]
[ExecuteInEditMode]
public class ConnectRect : MonoBehaviour
{
    public GameObject cude1;
    public GameObject cude2;
    Matrix4x4 cude1object2world, cude2object2world, cude1world2screen,cude2world2screen;
    public Camera cam1, cam2;
    Vector4 cude1frontbottomright, cude1fronttopright;
    Vector4 cude2frontbottomleft, cude2fronttopleft;
    Matrix4x4 cude1MVP;
    Matrix4x4 cude2MVP;
    // Start is called before the first frame update
    void Start()
    {
    }

    // Update is called once per frame
    void Update()
    {
        cude1object2world = cude1.transform.localToWorldMatrix;
        cude2object2world = cude2.transform.localToWorldMatrix;
        cude1world2screen = cam1.projectionMatrix;
        cude2world2screen = cam2.projectionMatrix;
        cude1MVP = cude1object2world * cude1world2screen;
        cude2MVP = cude2object2world * cude2world2screen;
        cude1frontbottomright = new Vector4(0.5f, 0.5f, 0.5f, 1);
        cude1fronttopright = new Vector4(0.5f, -0.5f, 0.5f, 1);
        cude2frontbottomleft = new Vector4(0.5f, 0.5f, 0.5f, 1);
        cude2fronttopleft = new Vector4(0.5f, -0.5f, 0.5f, 1);
        cude1frontbottomright = cude1MVP * cude1frontbottomright;
        cude1fronttopright = cude1MVP * cude1fronttopright;
        cude2frontbottomleft = cude2MVP * cude2frontbottomleft;
        cude2fronttopleft = cude2MVP * cude2fronttopleft;
        Debug.Log(cude1frontbottomright);
    }
}
