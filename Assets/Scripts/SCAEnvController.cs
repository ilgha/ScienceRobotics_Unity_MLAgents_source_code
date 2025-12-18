using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;

public class ShelterEnvController : MonoBehaviour
{
    public GameObject epuckPrefab; // Assign the Epuck prefab in the Inspector
    public GameObject shelterParent; // Assign the shelter GameObject in the Inspector
    public Light lightspot; // Assign the Light GameObject in the Inspector
    public int numberOfAgents = 10; // Number of agents to instantiate
    public Vector3 spawnAreaCenter = Vector3.zero; // Center of the spawn area
    public Vector3 spawnAreaSize = new Vector3(24f, 0f, 24); // Size of the spawn area

    public GameObject patch; // The floor patch (goal area)
    public int MaxEnvironmentSteps = 1800; // Fixed episode length

    private Vector3 patchCenter;
    private Vector3 patchSize; // Size of the rectangular patch
    private int stepCounter;
    private int simCounter;
    private SimpleMultiAgentGroup agentGroup;
    private List<Epuck> agentsList = new List<Epuck>();
    private float cumulReward = 0;

    void Start()
    {
        // Initialize patch center and size
        patchCenter = patch.transform.position;
        patchSize = patch.transform.localScale; // Assuming the patch is a rectangular plane

        // Initialize agent group
        agentGroup = new SimpleMultiAgentGroup();

        // Destroy any existing Epuck instances in the scene
        Epuck[] existingAgents = FindObjectsOfType<Epuck>();

        // Instantiate agents
        for (int i = 0; i < numberOfAgents; i++)
        {
            Vector3 spawnPos = GetRandomSpawnPos();
            GameObject agentObj = Instantiate(epuckPrefab, shelterParent.transform);
            agentObj.transform.localPosition = spawnPos;
            agentObj.transform.localRotation = Quaternion.Euler(0, Random.Range(0, 360), 0);
            Epuck agent = agentObj.GetComponent<Epuck>();
            agent.lightSource = lightspot;
            agentsList.Add(agent);
            agentGroup.RegisterAgent(agent);
        }

        ResetEnvironment();
        simCounter = 0;
    }

    void FixedUpdate()
    {
        stepCounter++;

        // Count robots in the patch
        int robotsInPatch = CountRobotsInPatch();
        float reward = robotsInPatch; // Reward proportional to the number of robots in the patch
        cumulReward += reward;

        // Assign the group reward
        agentGroup.AddGroupReward(reward);

        // End episode if max steps are reached
        if (stepCounter >= MaxEnvironmentSteps)
        {
            agentGroup.GroupEpisodeInterrupted();
            ResetEnvironment();
            simCounter++;
        }
    }

    void ResetEnvironment()
    {
        stepCounter = 0;
        cumulReward = 0;

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

    int CountRobotsInPatch()
    {
        int count = 0;
        foreach (var agent in agentsList)
        {
            Vector3 agentPosition = agent.transform.position;

            // Check if the agent is within the rectangular patch
            bool isInXBounds = Mathf.Abs(agentPosition.x - patchCenter.x)/10 <= patchSize.x / 2f;
            bool isInZBounds = Mathf.Abs(agentPosition.z - patchCenter.z)/10 <= patchSize.z / 2f;

            if (isInXBounds && isInZBounds)
            {
                count++;
            }
        }
        return count;
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
