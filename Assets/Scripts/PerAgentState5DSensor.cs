using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;

[RequireComponent(typeof(BehaviorParameters))]
public class PerAgentState5DSensor : SensorComponent
{
    [Header("Identification")]
    public string sensorName = "state5d";

    [Header("Arena (XZ)")]
    public Transform arenaCenter;    // set from your EnvController
    public float arenaRadius = 12f;  // normalization for r

    [Header("Mode")]
    public bool trainingOnly = true; // auto-disable if no trainer or a model is assigned

    [Header("Debug")]
    public bool verbose = true;
    public int printEveryNSteps = 120;

    private bool _active;
    private int _steps;

    public override ISensor[] CreateSensors()
    {
        var bp = GetComponent<BehaviorParameters>();
        bool hasModel = (bp != null && bp.Model != null);

        // In training Player builds, Model should be None, so stay ACTIVE.
        bool _active = !(trainingOnly && hasModel);

        if (!_active)
        {
            Debug.LogWarning($"[State5D] Disabled '{sensorName}' (trainingOnly={trainingOnly}) " +
                            $"because a model is assigned: {(hasModel ? bp.Model.name : "None")}");
            return new ISensor[0];
        }

        Debug.Log($"[State5D] '{sensorName}' ACTIVE (training). Model=None.");
        return new ISensor[] { new State5DISensor(this, sensorName) };
    }

    class State5DISensor : ISensor
    {
        private readonly PerAgentState5DSensor _owner;
        private readonly string _name;
        private readonly ObservationSpec _spec;

        public State5DISensor(PerAgentState5DSensor owner, string name)
        {
            _owner = owner; _name = name;
            _spec = ObservationSpec.Vector(5);
        }

        public ObservationSpec GetObservationSpec() => _spec;
        public string GetName() => _name;
        public CompressionSpec GetCompressionSpec() => CompressionSpec.Default();
        public byte[] GetCompressedObservation() => null;
        public void Reset() { }
        public void Update() { }

        public int Write(ObservationWriter writer)
        {
            _owner._steps++;

            Transform t = _owner.transform;
            Vector3 c = _owner.arenaCenter ? _owner.arenaCenter.position : Vector3.zero;

            Vector2 p = new Vector2(t.position.x - c.x, t.position.z - c.z);
            float norm = p.magnitude;
            float r = Mathf.Clamp01(norm / Mathf.Max(1e-6f, _owner.arenaRadius));

            float cosA = norm > 1e-6f ? (p.y / norm) : 1f; // z/r
            float sinA = norm > 1e-6f ? (p.x / norm) : 0f; // x/r

            Vector3 f3 = t.forward;
            Vector2 h = new Vector2(f3.x, f3.z).normalized;
            Vector2 rhat = norm > 1e-6f ? (p / norm) : new Vector2(0f, 1f);
            float cosB = Vector2.Dot(h, rhat);
            float sinB = rhat.x * h.y - rhat.y * h.x;

            writer[0] = r; writer[1] = cosA; writer[2] = sinA; writer[3] = cosB; writer[4] = sinB;

            if (_owner.verbose && (_owner._steps % Mathf.Max(1, _owner.printEveryNSteps) == 0))
            {
                Debug.Log($"[State5D] {_name} (r={r:F3}, cA={cosA:F3}, sA={sinA:F3}, cB={cosB:F3}, sB={sinB:F3}) on {_owner.name}");
            }
            return 5;
        }
    }
}
