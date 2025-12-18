using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;

public class DirectionalGateEnvController : MonoBehaviour
{
    public GameObject epuckPrefab; // Assign the Epuck prefab in the Inspector
    public GameObject arenaParent; // Assign the arena GameObject in the Inspector
    public Light lightspot; // Assign the Light GameObject in the Inspector
    public int numberOfAgents = 10; // Number of agents to instantiate
    public Vector3 spawnAreaCenter = Vector3.zero; // Center of the spawn area
    public Vector3 spawnAreaSize = new Vector3(24f, 0f, 24f); // Size of the spawn area

    public GameObject blackCorridor; // Assign the black corridor patch in the Inspector
    public GameObject whiteGate; // Assign the white gate patch in the Inspector
    public int MaxEnvironmentSteps = 1200; // Fixed episode length

    private int stepCounter;
    private int simCounter;
    private SimpleMultiAgentGroup agentGroup;
    private List<Epuck> agentsList = new List<Epuck>();
    private float cumulReward = 0;

    private int blackToWhiteCount = 0; // Number of robots passing black to white
    private int whiteToBlackCount = 0; // Number of robots passing white to black

    private Vector3 corridorCenter;
    private Vector3 corridorSize;
    private Vector3 gateCenter;
    private Vector3 gateSize;

    void Start()
    {
        // Initialize corridor and gate positions and sizes
        corridorCenter = blackCorridor.transform.position;
        corridorSize = blackCorridor.transform.localScale;
        gateCenter = whiteGate.transform.position;
        gateSize = whiteGate.transform.localScale;

        // Initialize agent group
        agentGroup = new SimpleMultiAgentGroup();

        // Instantiate agents
        for (int i = 0; i < numberOfAgents; i++)
        {
            Vector3 spawnPos = GetRandomSpawnPos();
            GameObject agentObj = Instantiate(epuckPrefab, arenaParent.transform);
            agentObj.transform.localPosition = spawnPos;
            agentObj.transform.localRotation = Quaternion.Euler(0, Random.Range(0, 360), 0);
            Epuck agent = agentObj.GetComponent<Epuck>();
            agent.lightSource = lightspot;
            agentsList.Add(agent);
            agentGroup.RegisterAgent(agent);

            var s5 = agentObj.GetComponent<PerAgentState5DSensor>();
            if (s5 == null)
            {
                Debug.LogError("[Env] Missing PerAgentState5DSensor on Epuck prefab! Please add it to the prefab.");
            }
            else
            {
                s5.sensorName   = "state5d";
                s5.arenaCenter  = (arenaParent != null) ? arenaParent.transform : null;
                s5.arenaRadius  = Mathf.Max(spawnAreaSize.x, spawnAreaSize.z) * 0.5f;
                s5.trainingOnly = true;         // you can temporarily set false to force it ON
                s5.verbose      = true;
                s5.printEveryNSteps = 120;
            }

            if (agentsList.Count == numberOfAgents)
            {
                string centerStr = (s5.arenaCenter != null)
                    ? s5.arenaCenter.position.ToString("F2")
                    : "(null -> using (0,0,0))";
                Debug.Log($"[Env] PerAgentState5DSensor attached to {agentsList.Count} agents. " +
                        $"center={centerStr}, radius={s5.arenaRadius:F2}, name='{s5.sensorName}'");
            }
        }

        ResetEnvironment();
        simCounter = 0;
    }

    void FixedUpdate()
    {
        // Debug.Log("TimeStep: " + stepCounter);
        stepCounter++;

        foreach (var agent in agentsList)
        {
            // Handle transitions
            if (agent.PreviousGroundColor == "black" && agent.CurrentGroundColor == "white")
            {
                blackToWhiteCount++;
                agentGroup.AddGroupReward(1.0f); // Reward for correct traversal
                cumulReward += 1.0f;
                Debug.Log($"Reward: " + cumulReward);
                //  Debug.Log($"Robot transitioned BLACK -> WHITE");
            }
            else if (agent.PreviousGroundColor == "white" && agent.CurrentGroundColor == "black")
            {
                whiteToBlackCount++;
                agentGroup.AddGroupReward(-1.0f); // Penalize reverse traversal
                cumulReward -= 1.0f;
                Debug.Log($"Reward: " + cumulReward);
                //  Debug.Log($"Robot transitioned WHITE -> BLACK");
            }
            else if (agent.PreviousGroundColor == "grey" && agent.CurrentGroundColor == "black")
            {
                //  Debug.Log($"Robot transitioned GREY -> BLACK");
            }
            else if (agent.PreviousGroundColor == "grey" && agent.CurrentGroundColor == "white")
            {
                //  Debug.Log($"Robot transitioned GREY -> WHITE");
            }
            else if (agent.PreviousGroundColor == "black" && agent.CurrentGroundColor == "grey")
            {
                //  Debug.Log($"Robot transitioned BLACK -> GREY");
            }
            else if (agent.PreviousGroundColor == "white" && agent.CurrentGroundColor == "grey")
            {
                //  Debug.Log($"Robot transitioned WHITE -> GREY");
            }

            // Update the robot's previous ground color
            agent.PreviousGroundColor = agent.CurrentGroundColor;
        }

        // End episode if max steps are reached
        if (stepCounter >= MaxEnvironmentSteps)
        {
            Debug.Log($"Reward: " + cumulReward);
            agentGroup.GroupEpisodeInterrupted();
            ResetEnvironment();
            simCounter++;
        }
    }

    void ResetEnvironment()
    {
        stepCounter = 0;
        cumulReward = 0;
        blackToWhiteCount = 0;
        whiteToBlackCount = 0;

        // Reset agents
        foreach (var agent in agentsList)
        {
            Vector3 spawnPos = GetRandomSpawnPos();
            agent.transform.localPosition = spawnPos;
            agent.transform.localRotation = Quaternion.Euler(0, Random.Range(0, 360), 0);
            Rigidbody rb = agent.GetComponent<Rigidbody>();
            rb.linearVelocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
            agent.EndEpisode();
        }
    }

    bool IsInCorridor(Vector3 position)
    {
        return Mathf.Abs(position.x - corridorCenter.x)/10 <= corridorSize.x / 2f &&
               Mathf.Abs(position.z - corridorCenter.z)/10 <= corridorSize.z / 2f;
    }

    bool IsInGate(Vector3 position)
    {
        return Mathf.Abs(position.x - gateCenter.x)/10 <= gateSize.x / 2f &&
               Mathf.Abs(position.z - gateCenter.z)/10 <= gateSize.z / 2f;
    }

    Vector3 GetRandomSpawnPos()
    {
        Vector3 randomPos = new Vector3(
            Random.Range(-spawnAreaSize.x / 2f, spawnAreaSize.x / 2f),
            0.5f, // Assuming agents are positioned slightly above the ground
            Random.Range(-spawnAreaSize.z / 2f, spawnAreaSize.z / 2f)
        );
        return spawnAreaCenter + randomPos;
    }
}
