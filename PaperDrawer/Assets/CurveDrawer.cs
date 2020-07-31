using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(LineRenderer))]
[ExecuteInEditMode]
public class CurveDrawer : MonoBehaviour
{
    public int segmentNum;
    public GameObject[] ControlObject;
    private Curver curver;
    private LineRenderer lineRenderer;
    // Start is called before the first frame update
    void Start()
    {
        curver = new Bezier();
        lineRenderer = GetComponent<LineRenderer>();
    }

    // Update is called once per frame
    void Update()
    {
        if (curver == null)
            curver = new Bezier();
        List<Vector3> position = new List<Vector3>();
        for (int i = 0; i < ControlObject.Length; i++)
        {
            position.Add(ControlObject[i].transform.position);
        }
        curver.SetControlPoint(position);
        var segments = curver.GetPosition(segmentNum);
        lineRenderer.positionCount = segments.Count;
        lineRenderer.SetPositions(segments.ToArray());
    }
}
public abstract class Curver
{
    public abstract void SetControlPoint(IEnumerable<Vector3> controlpoints);
    public abstract List<Vector3> GetPosition(int segmentNum);
}
//order = 3
public class Bezier : Curver
{
    private Vector3[] controlPoints = new Vector3[4];
    private List<Vector3> lastSegment;
    private int lastSegmentNum;
    private bool hasChangedAfterLast = true;
    public override List<Vector3> GetPosition(int segmentNum)
    {
        if (segmentNum == lastSegmentNum && !hasChangedAfterLast)
            return lastSegment;
        List<Vector3> r = new List<Vector3>();
        float t = 0;
        float dt = 1.0f / segmentNum;
        Vector3 now;
        for (int i = 0; i < segmentNum; i++) 
        {
            now = (1 - t) * (1 - t) * (1 - t) * controlPoints[0] +
                3 * t * (1 - t) * (1 - t) * controlPoints[1] +
                3 * t * t * (1 - t) * controlPoints[2] +
                t * t * t * controlPoints[3];
            r.Add(now);
            t += dt;
        }
        lastSegmentNum = segmentNum;
        hasChangedAfterLast = false;
        return r;
    }

    public override void SetControlPoint(IEnumerable<Vector3> controlpoints)
    {
        hasChangedAfterLast = true;
        int cpcount = 0;
        foreach (var item in controlpoints)
        {
            controlPoints[cpcount++] = item;
            if (cpcount == controlPoints.Length)
                break;
        }
    }
}
