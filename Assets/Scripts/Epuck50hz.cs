using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using System.Linq;

[RequireComponent(typeof(Rigidbody))]
[RequireComponent(typeof(BehaviorParameters))]
public class Epuck50hz : Agent
{
    [Header("Tick per seconds")]
    private float simulationElapsedTime = 0f; 
    private float controlTimer = 0f;
    public float controlUpdatePeriod = 0.1f; // 10 Hz control tick

    // For discrete controllers, store the current action:
    private int currentAction = 0;

    // For continuous controllers, you may store the desired wheel speeds:
    private float desiredLeftSpeed = 0f;
    private float desiredRightSpeed = 0f;
    // ----------------------------------------------------------------
    //  CHOOSE which type of NN controller we have:
    // ----------------------------------------------------------------
    [Header("Controller Type")]
    public ControllerType controllerType = ControllerType.Dandelion;

    // -------------------------------------------------------------
    //   NOISE (OPTIONAL)
    // -------------------------------------------------------------
    [Header("Noise Settings")]
    public bool enableNoise = true;
    // "epuck_wheels noise_std_dev" => wheelsNoiseStd
    [Range(0f, 1f)]
    public float wheelsNoiseStd = 0.05f;

    [Tooltip("Absolute noise level added to the behavior’s computed vector (e.g., 0.05)")]
    public float behaviorNoiseStd = 0.05f;

    [Tooltip("Absolute noise level for proximity sensor (e.g. 0.05)")]
    public float proxNoiseLevel = 0.05f;

    [Tooltip("Absolute noise level for light sensor (e.g. 0.05)")]
    public float lightNoiseLevel = 0.05f;

    [Tooltip("Absolute noise level for ground sensor (e.g. 0.05)")]
    public float groundNoiseLevel = 0.05f;

    [Header("RAB Noise Settings")]
    public float rabNoiseStd = 1.5f;          // Noise std deviation (absolute noise in distance)
    public float rabLossProbability = 0.85f;  // Probability to drop a message

    private float UniformNoise(float stdDev) {
        return Random.Range(-stdDev, stdDev);
    }
 
    // -------------------------------------------------------------
    //   ACTION SPACE
    // -------------------------------------------------------------
    [Header("Action Space Toggle")]
    public bool useContinuousActions = false;
 
    // -------------------------------------------------------------
    //   MOVEMENT & BODY
    // -------------------------------------------------------------
    private Rigidbody rBody;
    [Header("Differential Drive")]
    public float wheelBase = 0.55f;          // distance between e-puck wheels
    private float leftWheelVelocity  = 0f;
    private float rightWheelVelocity = 0f;
 
    [Header("Movement Speeds")]
    [Tooltip("Max wheel speed in m/s (e.g. 0.3 for ±30 cm/s)")]
    public float maxWheelSpeed = 0.16f;      // default ~16 cm/s
    // Smoothing factor for rotation (tunable)
    public float smoothingFactor = 1f;
 
    // -------------------------------------------------------------
    //   LAYERS & LINKS
    // -------------------------------------------------------------
    [Header("Sensors & Layers")]
    public LayerMask obstacleLayer;    // For proximity raycasts
    public LayerMask robotLayer;       // For RAB neighbor detection
    public Light lightSource;          // For the light sensor
    public bool CarryingFood { get; set; }
    public string PreviousGroundColor { get; set; } = "grey";
    public string CurrentGroundColor { get; set; } = "grey";
 
    // -------------------------------------------------------------
    //  PROXIMITY SENSOR (8 IR angles, single aggregated reading)
    // -------------------------------------------------------------
    [Header("Proximity Sensor")]
    [Tooltip("Typical e-puck IR range ~5cm in reality; adapt to your scene scale")]
    public float proxSensorRange = 0.1f;
 
    // e-puck IR sensor angles (from ARGoS code):
    private readonly float[] m_EpuckAnglesRad = {
        Mathf.PI / 10.5884f,   // ~17°
        Mathf.PI / 3.5999f,    // ~50°
        Mathf.PI / 2f,         // 90°
        Mathf.PI / 1.2f,       // 150°
        Mathf.PI / 0.8571f,    // 210°
        Mathf.PI / 0.6667f,    // 270°
        Mathf.PI / 0.5806f,    // 310°
        Mathf.PI / 0.5247f     // 342°
    };
 
    // Final single proximity reading
    public float ProximityValue  { get; private set; } // [0..1], clamp
    public float ProximityAngle  { get; private set; } // (-π..π)
    private float[] proxValues = new float[8];
 
    // -------------------------------------------------------------
    //  LIGHT SENSOR (8 angles, single aggregated reading)
    // -------------------------------------------------------------
    [Header("Light Sensor")]
    [Tooltip("If any sensor reading > 0.2 => final LightValue=1, else 0")]
    public float lightThreshold = 0.2f;
 
    public float LightValue { get; private set; } // 0 or 1
    public float LightAngle { get; private set; } // in radians (-π..π)
    private float[] lightValues = new float[8];
 
 
    // -------------------------------------------------------------
    //  GROUND SENSOR (3 IRs, queue of last 5 frames => final reading)
    // -------------------------------------------------------------
    [Header("Ground Sensor")]
    public float blackThreshold = 0.03f;
    public float whiteThreshold = 0.85f;
 
    public float[] groundSensor { get; private set; } = new float[3]; // [0]=black, [1]=white, [2]=grey
 
    // -------------------------------------------------------------
    //  RANGE & BEARING (time-limited neighbor buffer)
    // -------------------------------------------------------------
    [Header("Range & Bearing")]
    [Tooltip("Max detection radius for neighbors")]
    public float rabRange = 1.0f;
    public float alphaParameter = 5.0f; // used in Summation formula
 
    private class RabMessage {
        public int sourceID;
        public float ttl;
        public Vector3 position;
    }
 
    private List<RabMessage> rabBuffer = new List<RabMessage>();
    public int NumberNeighbors { get; private set; }
 
    // -------------------------------------------------------------
    //  DEBEGGING
    // -------------------------------------------------------------
    [Header("Debugging")]
    public bool debugSensors = false; // toggle this in the Inspector
 
    [Header("Debug Visuals")]
    public bool debugVisualSensors = false;
 
    // We'll store the raw data from each sensor update
    // so that in OnDrawGizmos() we can draw them.
    private struct ProximityRay {
        public Vector3 Origin;
        public Vector3 End;
        public bool Hit;
    }
    private ProximityRay[] m_ProxRays; // one entry per IR sensor
 
    private List<Vector3> m_GroundOffsetsWorld = new List<Vector3>();
    private List<Vector3> m_NeighborPositions = new List<Vector3>();
    private List<Vector3> m_RabNeighborPositions = new List<Vector3>();
 
    // We'll store the debug info for the light sensor rays
    private struct LightRay {
        public Vector3 Origin;
        public Vector3 End;
        public float Reading;
    }
 
    // This array will have 8 entries (one per light sensor angle)
    private LightRay[] m_LightRays;
 
    // A small scalar to make the rays visible in the Scene
    // (the reading might be <1, so we multiply by some distance)
    public float debugLightRayScale = 0.5f;
 
    // -------------------------------------------------------------
    //   ARGoS-AutoMoDe Behaviors
    // -------------------------------------------------------------
 
    // (1) STOP
    private void AutoMoDeStop() {
        // "ControlStep()" in ARGoS => m_pcRobotDAO->SetWheelsVelocity(0,0)
        leftWheelVelocity  = 0f;
        rightWheelVelocity = 0f;
        rBody.linearVelocity  = Vector3.zero;
        rBody.angularVelocity = Vector3.zero;
    }
 
    // (2) EXPLORATION: random walk + obstacle avoidance
    private enum ExplorationState { RANDOM_WALK, OBSTACLE_AVOIDANCE }
    private ExplorationState m_eExplorationState = ExplorationState.RANDOM_WALK;
    private int m_unTurnSteps = 0;
    // in ARGoS code, "m_fProximityThreshold=0.1" => we can use that or keep your frontProximityThreshold
    private float m_fProximityThreshold = 0.1f;
    // random steps range => "rwm" param in ARGoS. You can set a default or pass via param
    private Vector2Int m_cRandomStepsRange = new Vector2Int(1,5);  // e.g. [10..30]

    private class BehaviorState {
        public bool Avoiding = false;
        public int TurnSteps = 0;
        public TurnDirection TurnDir = TurnDirection.LEFT;
    }

    private BehaviorState m_sPhototaxisState  = new BehaviorState();
    private BehaviorState m_sAntiPhotoState   = new BehaviorState();
    private BehaviorState m_sAttractionState  = new BehaviorState();
    private BehaviorState m_sRepulsionState   = new BehaviorState();

    private void AutoMoDeInitExploration() {
        // Called once, like "Init()"
        m_unTurnSteps = 0;
        m_eExplorationState = ExplorationState.RANDOM_WALK;
        m_fProximityThreshold = 0.1f;  // from ARGoS
        // e.g. parse "rwm" param =>  m_cRandomStepsRange = (0..someMax)
    }
 
    private void AutoMoDeControlStepExploration() {
        switch (m_eExplorationState) {
            case ExplorationState.RANDOM_WALK:
                // Move forward at max speed.
                leftWheelVelocity = maxWheelSpeed;
                rightWheelVelocity = maxWheelSpeed;
                // Optionally add behavior noise to the forward command.
                if (enableNoise && behaviorNoiseStd > 0f) {
                    leftWheelVelocity += UniformNoise(behaviorNoiseStd);
                    rightWheelVelocity += UniformNoise(behaviorNoiseStd);
                    leftWheelVelocity = Mathf.Clamp(leftWheelVelocity, -maxWheelSpeed, maxWheelSpeed);
                    rightWheelVelocity = Mathf.Clamp(rightWheelVelocity, -maxWheelSpeed, maxWheelSpeed);
                }
                // Check if an obstacle is detected.
                if (IsObstacleInFront(ProximityValue, ProximityAngle, m_fProximityThreshold)) {
                    m_eExplorationState = ExplorationState.OBSTACLE_AVOIDANCE;
                    m_unTurnSteps = Random.Range(m_cRandomStepsRange.x, m_cRandomStepsRange.y);
                    m_eTurnDirection = (ProximityAngle < 0f) ? TurnDirection.LEFT : TurnDirection.RIGHT;
                }
                break;
            case ExplorationState.OBSTACLE_AVOIDANCE:
                m_unTurnSteps--;
                if (m_eTurnDirection == TurnDirection.LEFT) {
                    leftWheelVelocity = -maxWheelSpeed;
                    rightWheelVelocity = maxWheelSpeed;
                } else {
                    leftWheelVelocity = maxWheelSpeed;
                    rightWheelVelocity = -maxWheelSpeed;
                }
                if (m_unTurnSteps <= 0) {
                    m_eExplorationState = ExplorationState.RANDOM_WALK;
                }
                break;
        }
    }

 
    private bool IsObstacleInFront(float proxVal, float proxAngle, float threshold) {
        // ARGoS logic: if proxVal >= threshold and angle in [-π/2..+π/2]
        if (proxVal >= threshold) {
            if (proxAngle <= Mathf.PI*0.5f && proxAngle >= -Mathf.PI*0.5f) {
                return true;
            }
        }
        return false;
    }
 
    private enum TurnDirection { LEFT, RIGHT }
    private TurnDirection m_eTurnDirection;
 
    // (3) PHOTOTAXIS
    // sResultVector = sLightVector - 5*sProxVector
    private void AutoMoDeControlStepPhototaxis() {
        BehaviorState st = m_sPhototaxisState;
        if (st.Avoiding) {
            st.TurnSteps--;
            if (st.TurnDir == TurnDirection.LEFT) {
                leftWheelVelocity = -maxWheelSpeed;
                rightWheelVelocity = maxWheelSpeed;
            } else {
                leftWheelVelocity = maxWheelSpeed;
                rightWheelVelocity = -maxWheelSpeed;
            }
            if (st.TurnSteps <= 0)
                st.Avoiding = false;
        } else {
            if (IsObstacleInFront(ProximityValue, ProximityAngle, m_fProximityThreshold)) {
                st.Avoiding = true;
                st.TurnSteps = Random.Range(m_cRandomStepsRange.x, m_cRandomStepsRange.y);
                st.TurnDir = (ProximityAngle < 0f) ? TurnDirection.LEFT : TurnDirection.RIGHT;
            } else {
                // Compute the phototaxis vector.
                float lx = LightValue * Mathf.Cos(LightAngle);
                float ly = LightValue * Mathf.Sin(LightAngle);
                float px = ProximityValue * Mathf.Cos(ProximityAngle);
                float py = ProximityValue * Mathf.Sin(ProximityAngle);
                float rx = lx - .5f * px;
                float ry = ly - .5f * py;
                // Fallback if the computed vector is too small.
                float mag = Mathf.Sqrt(rx * rx + ry * ry);
                if (mag < 0.1f) { rx = 1f; ry = 0f; mag = 1f; }
                // Inject behavior noise.
                if (enableNoise && behaviorNoiseStd > 0f) {
                    rx += UniformNoise(behaviorNoiseStd);
                    ry += UniformNoise(behaviorNoiseStd);
                }
                (float L, float R) = ComputeWheelsVelocityFromVector(rx, ry, maxWheelSpeed);
                leftWheelVelocity = L;
                rightWheelVelocity = R;
            }
        }
    }
 
    // (4) ANTI-PHOTOTAXIS
    // sResultVector = -sLightVector - 5*sProxVector
    private void AutoMoDeControlStepAntiPhototaxis() {
        BehaviorState st = m_sAntiPhotoState;
        if (st.Avoiding) {
            st.TurnSteps--;
            if (st.TurnDir == TurnDirection.LEFT) {
                leftWheelVelocity = -maxWheelSpeed;
                rightWheelVelocity = maxWheelSpeed;
            } else {
                leftWheelVelocity = maxWheelSpeed;
                rightWheelVelocity = -maxWheelSpeed;
            }
            if (st.TurnSteps <= 0)
                st.Avoiding = false;
        } else {
            if (IsObstacleInFront(ProximityValue, ProximityAngle, m_fProximityThreshold)) {
                st.Avoiding = true;
                st.TurnSteps = Random.Range(m_cRandomStepsRange.x, m_cRandomStepsRange.y);
                st.TurnDir = (ProximityAngle < 0f) ? TurnDirection.LEFT : TurnDirection.RIGHT;
            } else {
                float lx = LightValue * Mathf.Cos(LightAngle);
                float ly = LightValue * Mathf.Sin(LightAngle);
                float px = ProximityValue * Mathf.Cos(ProximityAngle);
                float py = ProximityValue * Mathf.Sin(ProximityAngle);
                float rx = -lx - .5f * px;
                float ry = -ly - .5f * py;
                float mag = Mathf.Sqrt(rx * rx + ry * ry);
                if (mag < 0.1f) { rx = 1f; ry = 0f; mag = 1f; }
                if (enableNoise && behaviorNoiseStd > 0f) {
                    rx += UniformNoise(behaviorNoiseStd);
                    ry += UniformNoise(behaviorNoiseStd);
                }
                (float L, float R) = ComputeWheelsVelocityFromVector(rx, ry, maxWheelSpeed);
                leftWheelVelocity = L;
                rightWheelVelocity = R;
            }
        }
    }
 
    // (5) ATTRACTION
    // cRabReading => (range,bearing)
    // sResult = sRabVector - 6*sProxVector
    private void AutoMoDeControlStepAttraction() {
        // Get the attraction vector (range and bearing) from neighbors
        (float rabRange, float rabBearing) = GetAttractionVectorToNeighbors(alphaParameter);

        // Convert the attraction vector to x and y components.
        float rx = rabRange * Mathf.Cos(rabBearing);
        float ry = rabRange * Mathf.Sin(rabBearing);

        // Get the repulsion component from the proximity sensor.
        float px = ProximityValue * Mathf.Cos(ProximityAngle);
        float py = ProximityValue * Mathf.Sin(ProximityAngle);

        // Combine the attraction and repulsion signals.
        float rx2 = rx - .6f * px;
        float ry2 = ry - .6f * py;

        // If the resulting vector is too small, default to moving forward.
        float mag = Mathf.Sqrt(rx2 * rx2 + ry2 * ry2);
        if (mag < 0.1f) {
            rx2 = 1f;
            ry2 = 0f;
        }

        // Optionally add noise to the resultant vector.
        if (enableNoise && behaviorNoiseStd > 0f) {
            rx2 += UniformNoise(behaviorNoiseStd);
            ry2 += UniformNoise(behaviorNoiseStd);
        }

        // Compute wheel velocities from the combined vector.
        (float L, float R) = ComputeWheelsVelocityFromVector(rx2, ry2, maxWheelSpeed);
        leftWheelVelocity = L;
        rightWheelVelocity = R;
    }

 
    // (6) REPULSION
    // cRabReading => (range,bearing)
    // sResult = -repParam*sRabVector - 5*sProxVector
    //  (for simplicity, we assume alphaParameter is your repParam)
    private void AutoMoDeControlStepRepulsion() {
        // Get the neighbor-based (RAB) vector. Using the same function as attraction.
        (float rabRange, float rabBearing) = GetAttractionVectorToNeighbors(alphaParameter);
        
        // Convert the attraction vector to x and y components.
        float rx = rabRange * Mathf.Cos(rabBearing);
        float ry = rabRange * Mathf.Sin(rabBearing);
        
        // Get the repulsion component from the proximity sensor.
        float px = ProximityValue * Mathf.Cos(ProximityAngle);
        float py = ProximityValue * Mathf.Sin(ProximityAngle);
        
        // Combine the two components with the ARGoS scaling:
        // ARGoS: sResult = -ralphaParam * sRabVector - 5 * sProxVector
        float rx2 = -alphaParameter * rx - .5f * px;
        float ry2 = -alphaParameter * ry - .5f * py;
        
        float mag = Mathf.Sqrt(rx2 * rx2 + ry2 * ry2);
        if (mag < 0.1f) {
            rx2 = 1f;
            ry2 = 0f;
        }
        
        // Optionally add noise to the resultant vector.
        if (enableNoise && behaviorNoiseStd > 0f) {
            rx2 += UniformNoise(behaviorNoiseStd);
            ry2 += UniformNoise(behaviorNoiseStd);
        }
        
        // Compute wheel velocities from the resulting vector.
        (float L, float R) = ComputeWheelsVelocityFromVector(rx2, ry2, maxWheelSpeed);
        leftWheelVelocity = L;
        rightWheelVelocity = R;
    }

 
    /// <summary>
    ///  EXACT replication of AutoMoDe's "ComputeWheelsVelocityFromVector":
    ///  - Ignores magnitude of (x,y), only uses angle
    ///  - If angle in [0..π), Right=1, Left=cos(angle), else Right=cos(angle), Left=1
    ///  - Then scale to maxVel
    ///  - If (x,y) ~ (0,0) => wheels= (0,0)
    /// </summary>
    private (float left, float right) ComputeWheelsVelocityFromVector(float x, float y, float maxVel) {
        // 1) if vector near zero => (0,0)
        if(Mathf.Abs(x) < 1e-5f && Mathf.Abs(y) < 1e-5f) {
            return (0f, 0f);
        }

        // 2) get angle in [0..2π)
        float angle = Mathf.Atan2(y, x);   // range [-π..π]
        if(angle < 0f) angle += 2f*Mathf.PI; // force [0..2π)

        float left = 0f;
        float right= 0f;

        // 3) if angle in [0..π) => right=1, left=cos(angle)
        //    else => right=cos(angle), left=1
        if(angle < Mathf.PI) {
            // left hemisphere
            right = 1f;
            left  = Mathf.Cos(angle);
        } else {
            // right hemisphere
            right = Mathf.Cos(angle);
            left  = 1f;
        }

        // 4) scale them so that max(|left|, |right|) = 1 => then times maxVel
        float maxVal = Mathf.Max(Mathf.Abs(left), Mathf.Abs(right));
        if(maxVal < 1e-5f) {
            return (0f, 0f);
        }
        float scale = maxVel / maxVal;
        left  *= scale;
        right *= scale;

        return (left, right);
    }
 
    // -------------------------------------------------------------
    //   UNITY METHODS
    // -------------------------------------------------------------
    public override void Initialize()
    {
        rBody = GetComponent<Rigidbody>();
        rBody.constraints = RigidbodyConstraints.FreezeRotationX |
                            RigidbodyConstraints.FreezeRotationZ;
        rBody.linearDamping  = 10f;
        rBody.angularDamping = 10f;
 
        // Setup action space (continuous vs. discrete)
        var bp = GetComponent<BehaviorParameters>();
        // Decide the ActionSpec based on the chosen controller type:
        switch (controllerType)
        {
            case ControllerType.Dandelion:
                // 2 continuous (left wheel, right wheel):
                bp.BrainParameters.ActionSpec = ActionSpec.MakeContinuous(2);
                break;

            case ControllerType.Daisy:
                // 6 discrete actions (Stop, Exploration, Attraction, Repulsion, Phototaxis, AntiPhototaxis)
                bp.BrainParameters.ActionSpec = ActionSpec.MakeDiscrete(6);
                break;

            case ControllerType.Lilly:
                // 6 discrete actions again
                bp.BrainParameters.ActionSpec = ActionSpec.MakeDiscrete(6);
                break;
        }
    }
 
    public override void OnEpisodeBegin()
    {
        AutoMoDeStop();
        rabBuffer.Clear();
 
        // Update previous ground color if needed
        if (groundSensor[0] == 0f)      PreviousGroundColor = "black";
        else if (groundSensor[0] == 1f) PreviousGroundColor = "white";
        else if (groundSensor[0] == .5f) PreviousGroundColor = "grey";
    }
 
    // -------------------------------------------------------------
    //   COLLECT OBSERVATIONS: read sensors each step
    // -------------------------------------------------------------
    public override void CollectObservations(VectorSensor sensor)
    {
        // Update all sensors (like in ARGoS-based approach)
        UpdateProximitySensor();
        UpdateLightSensor();
        UpdateGroundSensor();
        UpdateRangeAndBearing();

        // Now decide which inputs to add, depending on the controller type:

        if (controllerType == ControllerType.Dandelion || controllerType == ControllerType.Daisy)
        {
            // 24 inputs (EvoStick style)

            // (1) 8 prox
            for (int i = 0; i < 8; i++)
                sensor.AddObservation(proxValues[i]);

            // (2) 8 light
            for (int i = 0; i < 8; i++)
                sensor.AddObservation(lightValues[i]);

            // (3) 3 ground
            for (int i = 0; i < 3; i++)
                sensor.AddObservation(groundSensor[i]);

            // (4) RAB
            // 4a) ztilde(n)
            float ztilde = GetZtilde(NumberNeighbors);
            sensor.AddObservation(ztilde);

            // 4b) Summation vector wr&b, then 4 directional projections
            float wrbX = 0f, wrbY = 0f;
            foreach (var msg in rabBuffer)
            {
                Vector3 diff = msg.position - transform.position;
                Vector3 localDir = transform.InverseTransformDirection(diff);
                float dist = localDir.magnitude;
                if (dist > 1e-5f)
                {
                    float bearing = Mathf.Atan2(localDir.x, localDir.z);
                    float invDist = 1f / dist;
                    float x = invDist * Mathf.Cos(bearing);
                    float y = invDist * Mathf.Sin(bearing);
                    wrbX += x; wrbY += y;
                }
            }
            float[] anglesDeg = { 45f, 135f, 225f, 315f };
            for (int i = 0; i < anglesDeg.Length; i++)
            {
                float rad = anglesDeg[i] * Mathf.Deg2Rad;
                float dx = Mathf.Cos(rad);
                float dy = Mathf.Sin(rad);
                float proj = wrbX * dx + wrbY * dy;
                sensor.AddObservation(proj);
            }
        }
        else if (controllerType == ControllerType.Lilly)
        {
            // only 4 inputs: groundSensor (3) + ztilde(n)
            for (int i = 0; i < 3; i++)
                sensor.AddObservation(groundSensor[i]);

            float ztilde = GetZtilde(NumberNeighbors);
            sensor.AddObservation(ztilde);
        }
    }
 
    // -------------------------------------------------------------
    //   ACTION LOGIC (CONTINUOUS or DISCRETE)
    // -------------------------------------------------------------
    public override void OnActionReceived(ActionBuffers actions)
    {
        // Debug.Log("Action Decided at: " + simulationElapsedTime);
        if (controllerType == ControllerType.Dandelion) // Continuous
        {
            // Cache desired wheel speeds (you may decide to update them immediately or later)
            desiredLeftSpeed = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f) * maxWheelSpeed;
            desiredRightSpeed = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f) * maxWheelSpeed;
        }
        else // Daisy or Lilly, discrete controllers
        {
            // Store the discrete action value
            currentAction = actions.DiscreteActions[0];
            // Debug.Log("Action: " + currentAction);
        }
    }
 
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // If we want, we can do a separate logic for each controller type
        if (controllerType == ControllerType.Dandelion)
        {
            var cont = actionsOut.ContinuousActions;
            float forward = 0f, turn = 0f;
            if (Input.GetKey(KeyCode.W)) forward = 1f;
            if (Input.GetKey(KeyCode.S)) forward = -1f;
            if (Input.GetKey(KeyCode.A)) turn = -1f;
            if (Input.GetKey(KeyCode.D)) turn = 1f;

            float left = forward - turn;
            float right = forward + turn;
            cont[0] = left;
            cont[1] = right;
        }
        else
        {
            // Daisy or Lilly => 6 discrete
            var disc = actionsOut.DiscreteActions;
            disc[0] = 0; // default Stop
            if (Input.GetKey(KeyCode.Q)) disc[0] = 1;
            if (Input.GetKey(KeyCode.W)) disc[0] = 2;
            if (Input.GetKey(KeyCode.E)) disc[0] = 3;
            if (Input.GetKey(KeyCode.R)) disc[0] = 4;
            if (Input.GetKey(KeyCode.T)) disc[0] = 5;
        }
    }

 
    // -------------------------------------------------------------
    //   FIXEDUPDATE: DIFFERENTIAL DRIVE KINEMATICS
    // -------------------------------------------------------------
    private void ExecuteControlStep() {
        if (controllerType == ControllerType.Dandelion)
        {
            // For continuous controllers, update the control (wheel speeds) using the cached desired speeds:
            leftWheelVelocity = desiredLeftSpeed;
            rightWheelVelocity = desiredRightSpeed;
        }
        else
        {
            // For discrete controllers, choose the behavior based on the currentAction value:
            switch (currentAction)
            {
                case 0:
                    AutoMoDeStop();
                    break;
                case 1:
                    AutoMoDeControlStepExploration();
                    break;
                case 2:
                    AutoMoDeControlStepAttraction();
                    break;
                case 3:
                    AutoMoDeControlStepRepulsion();
                    break;
                case 4:
                    AutoMoDeControlStepPhototaxis();
                    break;
                case 5:
                    AutoMoDeControlStepAntiPhototaxis();
                    break;
                default:
                    AutoMoDeStop();
                    break;
            }
        }
    }

    void FixedUpdate()
    {
        // Accumulate time for control (tick) updates.
        simulationElapsedTime += Time.fixedDeltaTime;
        controlTimer += Time.fixedDeltaTime;
        if (controlTimer >= controlUpdatePeriod)
        {
            ExecuteControlStep(); // update wheel speeds or behavior states based on stored actions
            controlTimer -= controlUpdatePeriod; // subtract the period (preserving any extra time)
        }

        // Physics integration: update position and rotation.
        float v = 0.5f * (leftWheelVelocity + rightWheelVelocity);
        float w = (rightWheelVelocity - leftWheelVelocity) / wheelBase;

        // Compute the target rotation based on the angular speed.
        Quaternion targetRotation = rBody.rotation * Quaternion.Euler(0f, w * Mathf.Rad2Deg * Time.fixedDeltaTime, 0f);
        // Interpolate current rotation for smoothness.
        Quaternion newRotation = Quaternion.Lerp(rBody.rotation, targetRotation, smoothingFactor);
        rBody.MoveRotation(newRotation);

        // Update position using constant speed; note multiplication by Time.fixedDeltaTime.
        Vector3 forward = newRotation * Vector3.forward;
        Vector3 newPosition = rBody.position + forward * v * Time.fixedDeltaTime;
        if (IsPathClear(newPosition))
            rBody.MovePosition(newPosition);
        else
            AutoMoDeStop();

        UpdateGroundColor();
        if (debugSensors)
            DebugSensors();
    }
 
    bool IsPathClear(Vector3 pos)
    {
        float radius = 0.1f;
        return !Physics.CheckSphere(pos, radius, obstacleLayer);
    }
 
    // -------------------------------------------------------------
    //   ARGoS-LIKE SENSORS
    // -------------------------------------------------------------
 
    // 1) PROXIMITY
    void UpdateProximitySensor()
    {
        Vector2 sumProx = Vector2.zero;

        if (m_ProxRays == null || m_ProxRays.Length != m_EpuckAnglesRad.Length)
        {
            m_ProxRays = new ProximityRay[m_EpuckAnglesRad.Length];
        }

        // For each IR sensor:
        for (int i = 0; i < m_EpuckAnglesRad.Length; i++)
        {
            float reading = 0f;  // Start with a baseline reading of 0

            float angleRad = m_EpuckAnglesRad[i] + Mathf.PI / 2f;
            Vector3 localDir3D = new Vector3(Mathf.Cos(angleRad), 0f, Mathf.Sin(angleRad));
            Vector3 worldDir = transform.TransformDirection(localDir3D);
            Vector3 origin = transform.position;

            // Combined mask: obstacles and other robots
            LayerMask combinedMask = obstacleLayer | robotLayer;
            bool hasHit = Physics.Raycast(origin, worldDir, out RaycastHit hit, proxSensorRange, combinedMask);

            m_ProxRays[i].Origin = origin;
            if (hasHit)
            {
                m_ProxRays[i].End = origin + worldDir * hit.distance;
                m_ProxRays[i].Hit = true;
                // Compute the reading from distance
                reading = 1f - (hit.distance / proxSensorRange);
            }
            else
            {
                m_ProxRays[i].End = origin + worldDir * proxSensorRange;
                m_ProxRays[i].Hit = false;
                // reading stays as 0 (no signal) initially
            }

            // Always add noise if enabled, regardless of hit or no hit.
            if (enableNoise && proxNoiseLevel > 0f)
            {
                float proxNoise = UniformNoise(proxNoiseLevel);
                //Debug.Log("Prox Noise: " + proxNoise);
                reading += proxNoise;
            }
            // Clamp the final reading to [0,1]
            reading = Mathf.Clamp01(reading);

            // Store the noisy reading
            proxValues[i] = reading;

            // Accumulate for net sensor direction (even if there was no hit, noise can drive a nonzero reading)
            sumProx += reading * new Vector2(localDir3D.z, localDir3D.x);
        }

        // Compute net magnitude and angle for proximity (used later in behavior control)
        float length = sumProx.magnitude;
        if (length > 1f) length = 1f;

        float angle = (length > 1e-9f) ? Mathf.Atan2(sumProx.y, sumProx.x) : 0f;

        ProximityValue = length;
        ProximityAngle = angle;
    }

    // 2) LIGHT
    void UpdateLightSensor()
    {
        if (m_LightRays == null || m_LightRays.Length != m_EpuckAnglesRad.Length) {
            m_LightRays = new LightRay[m_EpuckAnglesRad.Length];
        }

        Vector2 sumLight     = Vector2.zero;
        float   maxSensorVal = 0f;

        for (int i = 0; i < m_EpuckAnglesRad.Length; i++)
        {
            // Compute sensor's local + world direction
            float angleRad     = m_EpuckAnglesRad[i] + Mathf.PI / 2f;
            Vector3 localDir3D = new Vector3(Mathf.Cos(angleRad), 0f, Mathf.Sin(angleRad));
            Vector3 worldDir   = transform.TransformDirection(localDir3D);

            // Compute (possibly unclamped) reading
            float reading = ComputeLightReading(worldDir);
            lightValues[i] = Mathf.Clamp01(reading);

            if (enableNoise && lightNoiseLevel > 0f)
            {
                float noise = UniformNoise(lightNoiseLevel);
                lightValues[i] += noise;
                lightValues[i] = Mathf.Clamp01(lightValues[i]);
            }

            // Track maximum reading
            if (reading > maxSensorVal)
                maxSensorVal = reading;

            // Accumulate for net angle
            sumLight += reading * new Vector2(localDir3D.z, localDir3D.x);

            // Debug visualization
            m_LightRays[i].Origin  = transform.position;
            m_LightRays[i].Reading = reading;
            m_LightRays[i].End     = transform.position + worldDir * (reading * debugLightRayScale);
        }

        // Compute net direction
        float len = sumLight.magnitude;
        float netAngle = (len > 1e-9f) ? Mathf.Atan2(sumLight.y, sumLight.x) : 0f;

        // Optional threshold like in ARGoS: if maxVal <= 0.2 => 0 reading
        if (maxSensorVal > 0.2f) {
            LightValue = maxSensorVal;  // = obs[2] in ARGoS
            LightAngle = netAngle;      // = obs[3] in ARGoS
        } else {
            LightValue = 0f;
            LightAngle = 0f;
        }
    }

    /*****************************************
    * Simplified ComputeLightReading
    *****************************************/
    float ComputeLightReading(Vector3 sensorWorldDir)
    {
        // If no light source, no reading
        if (lightSource == null)
            return 0f;

        // Vector from robot to the light
        Vector3 toLight = lightSource.transform.position - transform.position;
        float dist    = toLight.magnitude * 1f;
        float distSqr = dist * dist;

        // Raw inverse-square intensity
        // float rawIntensity = lightSource.intensity / dist;

        // [Optional] clamp to [0, 1] if you want ARGoS-like 0..1 values
        // rawIntensity = Mathf.Clamp01(rawIntensity);

        // Zero out if angle > 90° from sensor direction
        // float angleDeg = Vector3.Angle(sensorWorldDir.normalized, toLight.normalized);
        // if(angleDeg > 90f) {
        //     rawIntensity = 0f;
        // }
        float dot = Vector3.Dot(sensorWorldDir.normalized, toLight.normalized);
        // If the sensor is facing away from the light (dot < 0), set that part to 0
        if (dot < 0f) dot = 0f;

        float rawIntensity = (lightSource.intensity / dist) * dot;

        // [Optional] add noise
        if(enableNoise && lightNoiseLevel > 0f) {
            float noise = UniformNoise(lightNoiseLevel);
            rawIntensity += noise;
            // rawIntensity = Mathf.Clamp01(rawIntensity); // (if you're using 0..1 range)
        }

        return rawIntensity;
    }
 
    // 3) GROUND
    void UpdateGroundSensor()
    {
        groundSensor[0] = 0;
        groundSensor[1] = 0;
        groundSensor[2] = 0;
    
        Ray ray = new Ray(transform.position, -transform.up);
        if (Physics.Raycast(ray, out RaycastHit hit, Mathf.Infinity))
        {
            Renderer hitRenderer = hit.collider.GetComponent<Renderer>();
            if (hitRenderer != null)
            {
                float grayscale = hitRenderer.material.color.grayscale;
    
                if (enableNoise && groundNoiseLevel > 0f)
                {
                    float noise = UniformNoise(groundNoiseLevel);
                    grayscale = Mathf.Clamp(grayscale + noise, 0f, 1f);
                }
    
                if (grayscale < 0.3f)
                {      
                    groundSensor[0] = 0f;
                    groundSensor[1] = 0f;
                    groundSensor[2] = 0f;   // DirGateDandel put 1
                }
                else if (grayscale > 0.7f)
                { 
                    groundSensor[0] = 1f;
                    groundSensor[1] = 1f;
                    groundSensor[2] = 1f;
                }
                else
                {                       
                    groundSensor[0] = .5f;
                    groundSensor[1] = .5f;
                    groundSensor[2] = .5f;
                }
            }
        }
    }
 
    void UpdateGroundColor()
    {
        if (groundSensor[0] == 0f)      CurrentGroundColor = "black";
        else if (groundSensor[0] == 1f) CurrentGroundColor = "white";
        else if (groundSensor[0] == .5f) CurrentGroundColor = "grey";
    }
 
    // 4) RANGE & BEARING
    void UpdateRangeAndBearing()
    {
        // 1) Clear old data
        rabBuffer.Clear();

        // 2) OverlapSphere for all robots in range
        Collider[] neighbors = Physics.OverlapSphere(transform.position, rabRange, robotLayer);
        foreach (var col in neighbors)
        {
            // skip if it's this robot
            if (col.gameObject == this.gameObject) 
                continue;

            // Attempt to get another Epuck script
            Epuck other = col.GetComponent<Epuck>();
            if (other == null) 
                continue;

            Vector3 diff = other.transform.position - transform.position;
            float dist   = diff.magnitude;
            if (dist < 1e-5f) 
                continue; // skip degenerate case

            // 2B) Check line-of-sight (cast a ray vs. obstacleLayer):
            Ray ray = new Ray(transform.position, diff.normalized);
            if (Physics.Raycast(ray, out RaycastHit hitInfo, dist, obstacleLayer))
            {
                // There's an obstacle in the way => skip neighbor
                continue;
            }

            // 2C) Packet loss check
            float roll = Random.value;
            if (roll < rabLossProbability)
            {
                // We lose the message => skip neighbor
                continue;
            }

            // 2D) Add uniform noise to the distance
            float noisyDist = dist;
            if (enableNoise && rabNoiseStd > 0f)
            {
                float noiseVal = UniformNoise(rabNoiseStd); // ± rabNoiseStd
                noisyDist += noiseVal;
                if (noisyDist < 0f) 
                    noisyDist = 0f;
            }

            // 2E) Construct the "noisy" neighbor position
            Vector3 noisyDiff = diff.normalized * noisyDist;
            Vector3 measuredPos = transform.position + noisyDiff;

            // 2F) Create & store an entry in rabBuffer
            RabMessage msg = new RabMessage();
            msg.sourceID = other.GetInstanceID();
            msg.ttl = 0f; // not used, but kept for structure
            msg.position = measuredPos;

            rabBuffer.Add(msg);
        }

        // 3) immediate neighbor count
        NumberNeighbors = rabBuffer.Count;
    }

    private float GetZtilde(int n)
    {
        // 1 - 2/(1 + e^n)
        float val = 1f - 2f / (1f + Mathf.Exp(n));
        // clamp to [-1,1], though normally not needed
        return Mathf.Clamp(val, -1f, 1f);
    }

    public float[] GetEvoStickInputs()
    {
        float[] inputs = new float[24];
        int idx = 0;

        // (1) 8 prox sensors
        for (int i = 0; i < 8; i++)
            inputs[idx++] = proxValues[i];  // already in [0..1]

        // (2) 8 light sensors
        for (int i = 0; i < 8; i++)
            inputs[idx++] = lightValues[i]; // in [0..1]

        // (3) 3 ground sensors
        for (int i = 0; i < 3; i++)
            inputs[idx++] = groundSensor[i]; // in {0,0.5,1}

        // (4) 5 RAB inputs

        // 4.1) \tilde{z}(n)
        float ztilde = GetZtilde(NumberNeighbors);
        inputs[idx++] = ztilde;

        // 4.2) RAB projections
        float wrbx = 0f, wrby = 0f;
        foreach (var msg in rabBuffer)
        {
            Vector3 diff = msg.position - transform.position;
            float dist = diff.magnitude;
            if (dist > 1e-5f)
            {
                float invDist = 1f/dist;
                Vector3 localDir = transform.InverseTransformDirection(diff);
                float bearing = Mathf.Atan2(localDir.x, localDir.z);

                float x = invDist * Mathf.Cos(bearing);
                float y = invDist * Mathf.Sin(bearing);
                wrbx += x;
                wrby += y;
            }
        }
        float[] angles = { 45f*Mathf.Deg2Rad, 135f*Mathf.Deg2Rad,
                        225f*Mathf.Deg2Rad,315f*Mathf.Deg2Rad };
        for (int i = 0; i < 4; i++) {
            float dx = Mathf.Cos(angles[i]);
            float dy = Mathf.Sin(angles[i]);
            float proj = wrbx*dx + wrby*dy;
            inputs[idx++] = proj; // optional clamp
        }

        return inputs; // length = 24
    }


    /* -------------------------------------------
    *  GET ATTRACTION VECTOR (like ARGoS code)
    * -------------------------------------------
    * Summation: for each neighbor => alpha/(1+dist)
    * final Range = length of sum, Bearing = angle
    */
    public (float Range, float Bearing) GetAttractionVectorToNeighbors(float alpha)
    {
        Vector2 sumVec = Vector2.zero;
        foreach (var msg in rabBuffer)
        {
            Vector3 diff = msg.position - transform.position;
            float dist    = diff.magnitude;
            float weight  = alpha / (1f + dist);
 
            // direction in 2D
            Vector3 localDir = transform.InverseTransformDirection(diff);
            float bearing = Mathf.Atan2(localDir.x, localDir.z);
            Vector2 vec2 = new Vector2(Mathf.Cos(bearing), Mathf.Sin(bearing)) * weight;
            sumVec += vec2;
        }
 
        float length = sumVec.magnitude;
        float ang    = 0f;
        if (length > 1e-9f)
            ang = Mathf.Atan2(sumVec.y, sumVec.x);

        return (length, ang);
    }
    void DebugSensors()
    {
        // 1) Gather the 24 EvoStick inputs
        float[] evoInputs = GetEvoStickInputs();

        // The array evoInputs has:
        //   - indices [0..7]   => 8 prox
        //   - indices [8..15]  => 8 light
        //   - indices [16..18] => 3 ground
        //   - index  [19]      => ztilde(n)
        //   - indices [20..23] => 4 RAB projections

        // 2) We can label them explicitly:
        //    Example: "prox0=0.12, prox1=0.00, ..."
        //    We'll store them in a temporary list of strings, then join them.

        var debugParts = new List<string>();

        // 2a) Proximity
        for (int i = 0; i < 8; i++)
        {
            debugParts.Add($"prox{i}={evoInputs[i]:F2}");
        }

        // 2b) Light
        for (int i = 8; i < 16; i++)
        {
            int li = i - 8;
            debugParts.Add($"light{li}={evoInputs[i]:F2}");
        }

        // 2c) Ground
        debugParts.Add($"ground0={evoInputs[16]:F2}");
        debugParts.Add($"ground1={evoInputs[17]:F2}");
        debugParts.Add($"ground2={evoInputs[18]:F2}");

        // 2d) ztilde
        debugParts.Add($"ztilde={evoInputs[19]:F2}");

        // 2e) 4 RAB projections
        debugParts.Add($"rab45={evoInputs[20]:F2}");
        debugParts.Add($"rab135={evoInputs[21]:F2}");
        debugParts.Add($"rab225={evoInputs[22]:F2}");
        debugParts.Add($"rab315={evoInputs[23]:F2}");

        // 3) Join them into a single line
        string debugLine = string.Join(", ", debugParts);

        // 4) Log it
        Debug.Log($"[Epuck Debug] EvoStickInputs => {debugLine}");
    }
 
    void OnDrawGizmos()
    {
        if (!debugVisualSensors) 
            return;

        DrawProximityGizmos();
        DrawLightGizmos();
        DrawRABRangeGizmos();
        DrawRABNeighborsGizmos();
        DrawRABSumVectorGizmos();
    }

    private void DrawProximityGizmos()
    {
        if (m_ProxRays == null) return;

        foreach (var ray in m_ProxRays)
        {
            // skip uninitialized
            if (ray.Origin == Vector3.zero && ray.End == Vector3.zero)
                continue;

            Gizmos.color = ray.Hit ? Color.red : Color.green;
            Gizmos.DrawLine(ray.Origin, ray.End);
            Gizmos.DrawSphere(ray.End, 0.005f);
        }
    }

    private void DrawLightGizmos()
    {
        if (m_LightRays == null) return;

        foreach (var lr in m_LightRays)
        {
            // color from black(0) to yellow(1)
            Color c = Color.Lerp(Color.black, Color.yellow, lr.Reading);
            Gizmos.color = c;

            Gizmos.DrawLine(lr.Origin, lr.End);
            Gizmos.DrawSphere(lr.End, 0.01f);
        }
    }

    private void DrawRABRangeGizmos()
    {
        // Just draw a wire sphere for the RAB detection range
        Gizmos.color = Color.blue;
        Gizmos.DrawWireSphere(transform.position, rabRange);
    }

    private void DrawRABNeighborsGizmos()
    {
        // If you want lines to each neighbor
        Gizmos.color = Color.cyan;
        foreach (var neighPos in m_RabNeighborPositions)
        {
            Gizmos.DrawLine(transform.position, neighPos);
            Gizmos.DrawSphere(neighPos, 0.02f);
        }
    }

    private void DrawRABSumVectorGizmos()
    {
        // Optionally draw the sum vector from GetAttractionVectorToNeighbors()
        (float sumRange, float sumBearing) = GetAttractionVectorToNeighbors(alphaParameter);
        if (sumRange > 1e-5f)
        {
            // Convert bearing to local xz-plane
            Vector3 sumDir = new Vector3(Mathf.Sin(sumBearing), 0f, Mathf.Cos(sumBearing));
            float debugScale = 1.0f; // how long you want the line
            Vector3 endPos = transform.position + transform.TransformDirection(sumDir) * (sumRange * debugScale);

            Gizmos.color = Color.magenta;
            Gizmos.DrawLine(transform.position, endPos);
            Gizmos.DrawSphere(endPos, 0.01f);
        }
    }

}
 
 