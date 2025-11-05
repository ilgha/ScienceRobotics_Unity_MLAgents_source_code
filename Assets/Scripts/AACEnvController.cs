using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;

public class AggregationEnvController : MonoBehaviour
{
    public GameObject epuckPrefab; // Assign the Epuck prefab in the Inspector
    public GameObject aacParent; // Assign the AAC GameObject in the Inspector
    public Light lightspot; // Assign the Light GameObject in the Inspector
    public int numberOfAgents = 10; // Number of agents to instantiate
    public Vector3 spawnAreaCenter = Vector3.zero; // Center of the spawn area
    public Vector3 spawnAreaSize = new Vector3(24f, 0f, 24f); // Size of the spawn area

    public GameObject patch; // The floor patch (goal area)
    public int MaxEnvironmentSteps = 1800; // Fixed episode length

    private Vector3 patchCenter;
    private float patchRadius;
    private int stepCounter;
    private int simCounter;
    private SimpleMultiAgentGroup agentGroup;
    private List<Epuck> agentsList = new List<Epuck>();
    private float cumulReward = 0;

    void Start()
    {
        // Initialize patch center and radius
        patchCenter = patch.transform.position;
        patchRadius = patch.transform.localScale.x / 2f; // Assuming uniform scaling and circular patch

        // Initialize agent group
        agentGroup = new SimpleMultiAgentGroup();

        // Destroy any existing Epuck instances in the scene
        Epuck[] existingAgents = FindObjectsOfType<Epuck>();
        // foreach (var agent in existingAgents)
        // {
        //     Destroy(agent.gameObject);
        // }

        // Instantiate agents
        for (int i = 0; i < numberOfAgents; i++)
        {
            Vector3 spawnPos = GetRandomSpawnPos();
            GameObject agentObj = Instantiate(epuckPrefab, aacParent.transform);
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
            float distanceToPatchCenter = Vector3.Distance(new Vector3(agentPosition.x, 0, agentPosition.z), new Vector3(patchCenter.x, 0, patchCenter.z));

            if (distanceToPatchCenter <= patchRadius)
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
