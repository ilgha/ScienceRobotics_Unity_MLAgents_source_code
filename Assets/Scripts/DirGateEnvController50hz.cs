using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;

public class DirectionalGateEnvController50hz : MonoBehaviour
{
    public GameObject epuckPrefab; // Assign the Epuck prefab in the Inspector
    public GameObject arenaParent; // Assign the arena GameObject in the Inspector
    public Light lightspot; // Assign the Light GameObject in the Inspector
    public int numberOfAgents = 10; // Number of agents to instantiate
    public Vector3 spawnAreaCenter = Vector3.zero; // Center of the spawn area
    public Vector3 spawnAreaSize = new Vector3(24f, 0f, 24f); // Size of the spawn area

    public GameObject blackCorridor; // Assign the black corridor patch in the Inspector
    public GameObject whiteGate; // Assign the white gate patch in the Inspector

    // Instead of using a fixed step count, we use a simulation timer.
    private float simulationElapsedTime = 0f; 
    public float simulationDuration = 120f; // 120 seconds for a 2-minute episode

    private int blackToWhiteCount = 0;
    private int whiteToBlackCount = 0;
    private Vector3 corridorCenter;
    private Vector3 corridorSize;
    private Vector3 gateCenter;
    private Vector3 gateSize;

    private SimpleMultiAgentGroup agentGroup;
    private List<Epuck> agentsList = new List<Epuck>();
    private float cumulReward = 0f;

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
        }

        ResetEnvironment();
    }

    void FixedUpdate()
    {
        // Accumulate fixed update time
        simulationElapsedTime += Time.fixedDeltaTime;

        // Process each agent—for example, reward assignments based on ground color transitions.
        foreach (var agent in agentsList)
        {
            if (agent.PreviousGroundColor == "black" && agent.CurrentGroundColor == "white")
            {
                blackToWhiteCount++;
                agentGroup.AddGroupReward(1.0f); 
                cumulReward += 1.0f;
            }
            else if (agent.PreviousGroundColor == "white" && agent.CurrentGroundColor == "black")
            {
                whiteToBlackCount++;
                agentGroup.AddGroupReward(-1.0f); 
                cumulReward -= 1.0f;
            }
            agent.PreviousGroundColor = agent.CurrentGroundColor;
        }

        // End the episode if the elapsed simulation time reaches the simulation duration
        if (simulationElapsedTime >= simulationDuration)
        {
            Debug.Log($"Total Reward: {cumulReward}");
            agentGroup.GroupEpisodeInterrupted();
            ResetEnvironment();
            simulationElapsedTime = 0f; // Reset the timer for the next episode
        }
    }

    void ResetEnvironment()
    {
        // Reset the environment by restarting the counters and repositioning agents.
        simulationElapsedTime = 0f;
        cumulReward = 0f;
        blackToWhiteCount = 0;
        whiteToBlackCount = 0;

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
