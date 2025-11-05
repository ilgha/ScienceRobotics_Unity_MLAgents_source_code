using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;

public class HomingEnvController : MonoBehaviour
{
    public GameObject epuckPrefab; // Assign the Epuck prefab in the Inspector
    public GameObject homingParent; // Assign the Homing GameObject in the Inspector
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
    private float arenaRadius = 8f; // Radius of the dodecagonal arena

    void Start()
    {
        // Initialize patch center and radius
        patchCenter = patch.transform.position;
        patchRadius = patch.transform.localScale.x / 2f; // Assuming uniform scaling and circular patch

        // Initialize agent group
        agentGroup = new SimpleMultiAgentGroup();

        // Destroy any existing Epuck instances in the scene
        Epuck[] existingAgents = FindObjectsOfType<Epuck>();

        // Instantiate agents
        for (int i = 0; i < numberOfAgents; i++)
        {
            Vector3 spawnPos = GetRandomSpawnPos();
            GameObject agentObj = Instantiate(epuckPrefab, homingParent.transform);
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

        // End episode if max steps are reached
        if (stepCounter >= MaxEnvironmentSteps)
        {
            int robotsInPatch = CountRobotsInPatch();
            Debug.Log($"Reward: " + robotsInPatch);
            agentGroup.AddGroupReward(robotsInPatch);
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
