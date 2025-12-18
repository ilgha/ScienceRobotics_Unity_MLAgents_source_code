using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;  // For BehaviorParameters

[RequireComponent(typeof(Rigidbody))]
[RequireComponent(typeof(BehaviorParameters))]
public class EpuckOld : Agent
{
    // -------------------------------------------------------------
    //   NOISE SETTINGS
    // -------------------------------------------------------------
    [Header("Noise Settings")]
    [Tooltip("If true, noise is applied to sensors and actuators.")]
    public bool enableNoise = true;

    [Tooltip("Fraction of the reading used to compute uniform noise range, e.g. 0.05 = ±5%.")]
    [Range(0f, 1f)]
    public float sensorNoisePercent = 0.05f;

    [Tooltip("Fraction of the actuator command used for uniform noise, e.g. 0.05 = ±5%.")]
    [Range(0f, 1f)]
    public float actuatorNoisePercent = 0.05f;

    // ------------------------------
    //   INSPECTOR / TOGGLE
    // ------------------------------
    [Header("Action Space Toggle")]
    [Tooltip("Check this to use 2 continuous actions (raw wheel velocities).\n" +
             "Uncheck to use 1 discrete action (6 possible behaviors).")]
    public bool useContinuousActions = false;

    // ------------------------------
    //  RIGIDBODY & WHEEL MODEL
    // ------------------------------
    private Rigidbody rBody;
    public float wheelBase = 0.55f; // distance between wheels

    // Wheel velocities
    private float leftWheelVelocity  = 0f;
    private float rightWheelVelocity = 0f;

    // A maximum speed to scale our raw continuous actions
    [Header("Movement Speeds")]
    public float maxWheelSpeed = 0.16f;
    public float moveSpeed     = 0.16f;

    // ------------------------------
    //        LIGHT & SENSORS
    // ------------------------------
    public Light lightSource;
    public float sensorRange = 1f;
    public LayerMask obstacleLayer;
    public LayerMask robotLayer;
    public bool CarryingFood { get; set; }
    public string PreviousGroundColor { get; set; } = "grey"; 
    public string CurrentGroundColor { get; set; } = "grey";

    private float[] proximitySensors         = new float[4];
    private float[] lightSensors             = new float[4];
    private float[] groundSensor             = new float[3];
    private Vector3 averageNeighborPosition  = Vector3.zero;
    private float neighborDensity            = 0f;
    private int neighborCount                = 0;

    // ------------------------------------
    //      OBSTACLE AVOIDANCE HELPERS
    // ------------------------------------
    public int frontSensorIndex = 0;  
    public float frontSensorThreshold = 1f; 
    public int minRotateSteps = 10;     
    public int maxRotateSteps = 20;     
    private int rotateStepsRemaining = 0; 

    // ------------------------------------
    //  INITIAL SETUP: OVERRIDE ACTION SPEC
    // ------------------------------------
    public override void Initialize()
    {
        rBody = GetComponent<Rigidbody>();
        rBody.constraints = RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ;
        rBody.linearDamping = 10f; 
        rBody.angularDamping = 10f;

        // Grab BehaviorParameters on this agent
        var bp = GetComponent<BehaviorParameters>();

        // If we want continuous wheel velocities...
        if (useContinuousActions)
        {
            // 2 continuous actions => left & right wheel
            bp.BrainParameters.ActionSpec = ActionSpec.MakeContinuous(2);
        }
        else
        {
            // 1 discrete branch with 6 possible actions
            bp.BrainParameters.ActionSpec = ActionSpec.MakeDiscrete(6);
        }
    }

    // ------------------------------------
    //       RESET EPISODE
    // ------------------------------------
    public override void OnEpisodeBegin()
    {
        // Reset wheel velocities
        leftWheelVelocity      = 0f;
        rightWheelVelocity     = 0f;
        rotateStepsRemaining   = 0;

        // Update previous ground color if needed
        if (groundSensor[0] > 0)      PreviousGroundColor = "black"; 
        else if (groundSensor[1] > 0) PreviousGroundColor = "white"; 
        else if (groundSensor[2] > 0) PreviousGroundColor = "grey";
    }

    // ------------------------------------
    //     OBSERVATIONS
    // ------------------------------------
    public override void CollectObservations(VectorSensor sensor)
    {
        ReadProximitySensors();
        ReadLightSensors();
        ReadGroundSensor();
        ReadRangeAndBearingSensors();

        // Proximity (4)
        for (int i = 0; i < proximitySensors.Length; i++)
            sensor.AddObservation(proximitySensors[i]);

        // Light (4)
        for (int i = 0; i < lightSensors.Length; i++)
            sensor.AddObservation(lightSensors[i]);

        // Ground (3)
        for (int i = 0; i < groundSensor.Length; i++)
            sensor.AddObservation(groundSensor[i]);

        // Range/Bearing (position difference, scalar density)
        sensor.AddObservation(averageNeighborPosition - transform.position);
        sensor.AddObservation(neighborDensity);
    }

    // ------------------------------------
    //   ON ACTION RECEIVED: HANDLE DISCRETE OR CONTINUOUS
    // ------------------------------------
    public override void OnActionReceived(ActionBuffers actions)
    {
        if (useContinuousActions)
        {
            // TWO continuous actions for raw left/right wheel velocities
            float left  = actions.ContinuousActions[0];
            float right = actions.ContinuousActions[1];

            // Scale them by maxWheelSpeed
            leftWheelVelocity  = Mathf.Clamp(left, -1f, 1f) * maxWheelSpeed;
            rightWheelVelocity = Mathf.Clamp(right, -1f, 1f) * maxWheelSpeed;

            // --- NOISE: Add uniform noise to the raw actuator commands
            if (enableNoise && actuatorNoisePercent > 0f)
            {
                // Left wheel noise
                float leftRange = actuatorNoisePercent * leftWheelVelocity;
                float leftNoise = Random.Range(-leftRange, leftRange);
                leftWheelVelocity += leftNoise;

                // Right wheel noise
                float rightRange = actuatorNoisePercent * rightWheelVelocity;
                float rightNoise = Random.Range(-rightRange, rightRange);
                rightWheelVelocity += rightNoise;

                // Optionally clamp again so we don't exceed ±maxWheelSpeed
                leftWheelVelocity  = Mathf.Clamp(leftWheelVelocity, -maxWheelSpeed, maxWheelSpeed);
                rightWheelVelocity = Mathf.Clamp(rightWheelVelocity, -maxWheelSpeed, maxWheelSpeed);
            }
        }
        else
        {
            // ONE discrete action in [0..5], corresponding to your 6 behaviors
            int action = actions.DiscreteActions[0];
            switch (action)
            {
                case 0: Stop();       break;
                case 1: Explore();    break;
                case 2: Attract();    break;
                case 3: Repel();      break;
                case 4: Phototaxis(); break;
                case 5: AntiPhototaxis(); break;
            }
        }
    }

    // ------------------------------------
    //  HEURISTIC: ALSO SWITCH MODES
    // ------------------------------------
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        if (useContinuousActions)
        {
            // Fill continuous actions array
            var contActions = actionsOut.ContinuousActions;
            
            // Example: simple keyboard control 
            //  W = forward, S = backward
            //  A = turn left, D = turn right
            float forward = 0f;
            float turn = 0f;
            if (Input.GetKey(KeyCode.W)) forward = 1f;
            if (Input.GetKey(KeyCode.S)) forward = -1f;
            if (Input.GetKey(KeyCode.A)) turn = -1f;
            if (Input.GetKey(KeyCode.D)) turn = 1f;

            // Simple differential drive
            float left = forward - turn;
            float right = forward + turn;

            contActions[0] = left;
            contActions[1] = right;
        }
        else
        {
            // Fill discrete action (only 1 branch)
            var discActions = actionsOut.DiscreteActions;
            discActions[0] = 0; // default to Stop

            // For example:
            // Q => Explore   (1)
            // W => Attract   (2)
            // E => Repel     (3)
            // R => Phototaxis (4)
            // T => AntiPhototaxis (5)
            if (Input.GetKey(KeyCode.Q)) discActions[0] = 1;
            else if (Input.GetKey(KeyCode.W)) discActions[0] = 2;
            else if (Input.GetKey(KeyCode.E)) discActions[0] = 3;
            else if (Input.GetKey(KeyCode.R)) discActions[0] = 4;
            else if (Input.GetKey(KeyCode.T)) discActions[0] = 5;
        }
    }

    // ------------------------------------
    //       FIXED UPDATE & KINEMATICS
    // ------------------------------------
    void FixedUpdate()
    {
        // Standard differential-drive forward & turn
        float v = 0.5f * (leftWheelVelocity + rightWheelVelocity);
        float w = (rightWheelVelocity - leftWheelVelocity) / wheelBase;

        Quaternion rotation = Quaternion.Euler(0f, w * Mathf.Rad2Deg, 0f);
        rBody.MoveRotation(rBody.rotation * rotation);

        Vector3 forward = rBody.rotation * Vector3.forward;
        Vector3 newPosition = rBody.position + forward * v;

        // Ensure path is not blocked
        if (IsPositionClear(newPosition))
        {
            rBody.MovePosition(newPosition);
        }
        else
        {
            Stop(); // path blocked => stop
        }

        UpdateGroundColor();
    }

    // ------------------------------------
    //            SENSOR READS
    // ------------------------------------
    void ReadProximitySensors()
    {
        for (int i = 0; i < 4; i++)
        {
            Vector3 direction = Quaternion.Euler(0, i * 90, 0) * transform.forward;
            Ray ray = new Ray(transform.position, direction);
            int combinedMask = obstacleLayer | robotLayer;

            if (Physics.Raycast(ray, out RaycastHit hit, sensorRange, combinedMask))
            {
                float ratio = hit.distance / sensorRange;
                proximitySensors[i] = 1f - ratio; // closer => higher
            }
            else
            {
                proximitySensors[i] = 0f;
            }

            // --- NOISE: Add uniform noise to sensor reading
            if (enableNoise && sensorNoisePercent > 0f)
            {
                float val = proximitySensors[i];
                float noiseRange = sensorNoisePercent * val;
                float noise = Random.Range(-noiseRange, noiseRange);
                proximitySensors[i] = val + noise;
                proximitySensors[i] = Mathf.Clamp(proximitySensors[i], 0f, 1f);
            }
        }
    }

    void ReadLightSensors()
    {
        for (int i = 0; i < 4; i++)
        {
            lightSensors[i] = 0f;

            Vector3 sensorDirection = Quaternion.Euler(0, i * 90, 0) * transform.forward;
            Vector3 toLight = lightSource.transform.position - transform.position;
            float distance = toLight.magnitude;

            if (distance <= lightSource.range)
            {
                float intensityAtSensor = lightSource.intensity / (distance * distance);
                float angle = Vector3.Angle(sensorDirection, toLight.normalized);
                if (angle < 45f)
                {
                    lightSensors[i] += Mathf.Clamp(intensityAtSensor, 0f, 1f);
                }
            }

            // --- NOISE: Add uniform noise to light sensor
            if (enableNoise && sensorNoisePercent > 0f)
            {
                float val = lightSensors[i];
                float noiseRange = sensorNoisePercent * val;
                float noise = Random.Range(-noiseRange, noiseRange);
                lightSensors[i] = val + noise;
                lightSensors[i] = Mathf.Clamp(lightSensors[i], 0f, 1f);
            }
        }
    }

    void ReadGroundSensor()
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

                // --- NOISE: Add uniform noise to ground sensor
                if (enableNoise && sensorNoisePercent > 0f)
                {
                    float val = grayscale;
                    float noiseRange = sensorNoisePercent * val;
                    float noise = Random.Range(-noiseRange, noiseRange);
                    grayscale = val + noise;
                    grayscale = Mathf.Clamp(grayscale, 0f, 1f);
                }

                if (grayscale < 0.3f)      groundSensor[0] = 1f;
                else if (grayscale > 0.7f) groundSensor[1] = 1f;
                else                       groundSensor[2] = 1f;
            }
        }
    }

    void ReadRangeAndBearingSensors()
    {
        Collider[] neighbors = Physics.OverlapSphere(transform.position, sensorRange, robotLayer);
        neighborCount = 0;
        averageNeighborPosition = Vector3.zero;

        neighborDensity = 1f - (2f / (1f + Mathf.Exp(neighbors.Length)));

        foreach (Collider neighbor in neighbors)
        {
            if (neighbor.gameObject != gameObject)
            {
                neighborCount++;
                averageNeighborPosition += neighbor.transform.position;
            }
        }

        if (neighborCount > 0)
            averageNeighborPosition /= neighborCount;
        else
            averageNeighborPosition = transform.position;

        // --- NOISE: neighborDensity or averageNeighborPosition
        if (enableNoise && sensorNoisePercent > 0f)
        {
            // Add small positional noise.
            float noiseMag = sensorNoisePercent * sensorRange; 
            Vector3 noiseVec = new Vector3(Random.Range(-noiseMag, noiseMag), 0f, Random.Range(-noiseMag, noiseMag));
            averageNeighborPosition += noiseVec;

            // For neighborDensity, clamp to [0..1]
            float densityNoiseRange = sensorNoisePercent * neighborDensity;
            float densityNoise = Random.Range(-densityNoiseRange, densityNoiseRange);
            neighborDensity = Mathf.Clamp(neighborDensity + densityNoise, 0f, 1f);
        }
    }

    void UpdateGroundColor()
    {
        if (groundSensor[0] > 0)      CurrentGroundColor = "black"; 
        else if (groundSensor[1] > 0) CurrentGroundColor = "white"; 
        else if (groundSensor[2] > 0) CurrentGroundColor = "grey";
    }

    // ------------------------------------
    //        MOVEMENT HELPERS
    // ------------------------------------
    bool IsPositionClear(Vector3 position)
    {
        float clearanceRadius = 0.1f;
        return !Physics.CheckSphere(position, clearanceRadius, obstacleLayer);
    }

    void Stop()
    {
        leftWheelVelocity = 0f;
        rightWheelVelocity = 0f;
        rBody.linearVelocity = Vector3.zero;
        rBody.angularVelocity = Vector3.zero;
    }

    // -------------- DISCRETE BEHAVIORS --------------
    void Explore()
    {
        if (rotateStepsRemaining > 0)
        {
            rotateStepsRemaining--;
            leftWheelVelocity  = -moveSpeed;
            rightWheelVelocity =  moveSpeed;
            return;
        }

        float frontVal = proximitySensors[frontSensorIndex];
        if (frontVal > frontSensorThreshold)
        {
            rotateStepsRemaining = Random.Range(minRotateSteps, maxRotateSteps);
            Stop();
            return;
        }

        leftWheelVelocity  = moveSpeed;
        rightWheelVelocity = moveSpeed;
    }

    void Attract()
    {
        if (neighborCount > 0)
        {
            Vector3 dir = (averageNeighborPosition - transform.position).normalized;
            MoveTowards(dir);
        }
        else
        {
            Explore();
        }
    }

    void Repel()
    {
        if (neighborCount > 0)
        {
            Vector3 dir = (transform.position - averageNeighborPosition).normalized;
            MoveTowards(dir);
        }
        else
        {
            Explore();
        }
    }

    void Phototaxis()
    {
        if (lightSource.intensity <= 0f)
        {
            Explore();
            return;
        }

        Vector3 toLight = lightSource.transform.position - transform.position;
        if (proximitySensors[frontSensorIndex] > frontSensorThreshold)
        {
            AvoidObstacle();
            return;
        }

        if (toLight.magnitude > 0.1f)
        {
            DriveTowards(toLight, moveSpeed);
        }
        else
        {
            Stop();
        }
    }

    void AntiPhototaxis()
    {
        if (lightSource.intensity <= 0f)
        {
            Explore();
            return;
        }

        Vector3 away = transform.position - lightSource.transform.position;
        if (proximitySensors[frontSensorIndex] > frontSensorThreshold)
        {
            AvoidObstacle();
            return;
        }

        if (away.magnitude > 0.1f)
        {
            DriveTowards(away, moveSpeed);
        }
        else
        {
            Stop();
        }
    }

    // -------------- COMMON MOVEMENT UTILS --------------
    void DriveTowards(Vector3 direction, float baseSpeed)
    {
        direction.y = 0f;
        direction.Normalize();

        float angle = Vector3.SignedAngle(transform.forward, direction, Vector3.up);
        float turnFactor = Mathf.Clamp(angle / 90f, -1f, 1f);
        float leftVel  = baseSpeed * (1f - turnFactor);
        float rightVel = baseSpeed * (1f + turnFactor);

        leftWheelVelocity  = leftVel;
        rightWheelVelocity = rightVel;
    }

    void MoveTowards(Vector3 direction)
    {
        float dist = direction.magnitude;
        if (dist < 0.1f)
        {
            Stop();
            return;
        }
        DriveTowards(direction.normalized, moveSpeed);
    }

    void AvoidObstacle()
    {
        if (rotateStepsRemaining > 0)
        {
            rotateStepsRemaining--;
            leftWheelVelocity  = -moveSpeed;
            rightWheelVelocity =  moveSpeed;
        }
        else
        {
            rotateStepsRemaining = Random.Range(minRotateSteps, maxRotateSteps);
        }
    }
}
