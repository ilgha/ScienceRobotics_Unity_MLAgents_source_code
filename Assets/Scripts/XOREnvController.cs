using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;

public class XOREnvController : MonoBehaviour
{
    public GameObject epuckPrefab; // Assign the Epuck prefab in the Inspector
    public GameObject xorParent; // Assign the XOR GameObject in the Inspector
    public Light lightspot; // Assign the Light GameObject in the Inspector
    public int numberOfAgents = 10; // Number of agents to instantiate
    public Vector3 spawnAreaCenter = Vector3.zero; // Center of the spawn area
    public Vector3 spawnAreaSize = new Vector3(24f, 0f, 24f); // Size of the spawn area

    public List<GameObject> blackPatches; // List of black patches in the environment
    public int MaxEnvironmentSteps = 1800; // Fixed episode length

    private int stepCounter;
    private int simCounter;
    private SimpleMultiAgentGroup agentGroup;
    private List<Epuck> agentsList = new List<Epuck>();

    private float arenaRadius = 8f; // Radius of the dodecagonal arena

    void Start()
    {
        // Initialize agent group
        agentGroup = new SimpleMultiAgentGroup();

        // Instantiate agents
        for (int i = 0; i < numberOfAgents; i++)
        {
            Vector3 spawnPos = GetRandomSpawnPos();
            GameObject agentObj = Instantiate(epuckPrefab, xorParent.transform);
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

        // Count robots in patches and calculate reward
        int[] robotsInPatches = CountRobotsInPatches();
        int majorityCount = Mathf.Max(robotsInPatches); // Majority robot count
        float reward = majorityCount; // Reward proportional to majority count

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

    int[] CountRobotsInPatches()
    {
        int[] counts = new int[blackPatches.Count];

        foreach (var agent in agentsList)
        {
            Vector3 agentPosition = agent.transform.position;

            for (int i = 0; i < blackPatches.Count; i++)
            {
                Vector3 patchCenter = blackPatches[i].transform.position;
                float patchRadius = blackPatches[i].transform.localScale.x / 2f; // Assuming circular patches

                float distanceToPatchCenter = Vector3.Distance(new Vector3(agentPosition.x, 0, agentPosition.z), new Vector3(patchCenter.x, 0, patchCenter.z));
                if (distanceToPatchCenter <= patchRadius)
                {
                    counts[i]++;
                }
            }
        }

        return counts;
    }

    Vector3 GetRandomSpawnPos()
    {
        Vector3 spawnPos;
        int maxAttempts = 100;
        int attempts = 0;

        do
        {
            // Generate a random position within the rectangular area
            Vector3 randomPos = new Vector3(
                Random.Range(-spawnAreaSize.x / 2f, spawnAreaSize.x / 2f),
                0.5f, // Assuming agents are positioned slightly above the ground
                Random.Range(-spawnAreaSize.z / 2f, spawnAreaSize.z / 2f)
            );
            spawnPos = spawnAreaCenter + randomPos;

            // Check if the position is within the dodecagonal arena
            attempts++;
        } while (!IsInsideDodecagon(spawnPos, arenaRadius) && attempts < maxAttempts);

        if (attempts >= maxAttempts)
        {
            Debug.LogWarning("Failed to find a valid spawn position after maximum attempts.");
        }

        return spawnPos;
    }

    bool IsInsideDodecagon(Vector3 position, float radius)
    {
        // Check if the position is within a circular approximation of the dodecagon
        Vector3 centerToPosition = position - spawnAreaCenter;
        float distance = new Vector2(centerToPosition.x, centerToPosition.z).magnitude;
        return distance <= radius;
    }
}
