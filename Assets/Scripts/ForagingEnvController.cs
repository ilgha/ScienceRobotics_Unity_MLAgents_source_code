using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;

public class ForagingEnvController : MonoBehaviour
{
    public GameObject epuckPrefab; // Assign the Epuck prefab in the Inspector
    public GameObject foragingParent; // Assign the foraging GameObject in the Inspector
    public Light lightspot; // Assign the Light GameObject in the Inspector
    public int numberOfAgents = 10; // Number of agents to instantiate
    public Vector3 spawnAreaCenter = Vector3.zero; // Center of the spawn area
    public Vector3 spawnAreaSize = new Vector3(24f, 0f, 24f); // Size of the spawn area

    public List<GameObject> foodSources; // Assign black patches (food areas) in the Inspector
    public GameObject nest; // Assign the white patch (nest) in the Inspector
    public int MaxEnvironmentSteps = 1800; // Fixed episode length

    private int stepCounter;
    private int simCounter;
    private SimpleMultiAgentGroup agentGroup;
    private List<Epuck> agentsList = new List<Epuck>();
    private float cumulReward = 0;

    void Start()
    {
        // Initialize agent group
        agentGroup = new SimpleMultiAgentGroup();

        // Instantiate agents
        for (int i = 0; i < numberOfAgents; i++)
        {
            Vector3 spawnPos = GetRandomSpawnPos();
            GameObject agentObj = Instantiate(epuckPrefab, foragingParent.transform);
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

        // Reward robots for reaching the nest or food
        foreach (var agent in agentsList)
        {
            Vector3 agentPos = agent.transform.position;

            // Check if the robot is in the nest or food source
            if (IsInNest(agentPos) && agent.CarryingFood)
            {
                // Reward for returning food to the nest
                //agent.AddReward(1.0f);
                agentGroup.AddGroupReward(1.0f);
                cumulReward += 1.0f;
                agent.CarryingFood = false; // Drop the food
            }
            else if (IsInFoodSource(agentPos) && !agent.CarryingFood)
            {
                // Reward for collecting food
                // agent.AddReward(0.5f);
                // cumulReward += 0.5f;
                agent.CarryingFood = true; // Pick up the food
            }
        }

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
            agent.CarryingFood = false; // Reset food state
            Rigidbody rb = agent.GetComponent<Rigidbody>();
            rb.linearVelocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
            agent.EndEpisode();
        }
    }

    bool IsInNest(Vector3 position)
    {
        Vector3 nestCenter = nest.transform.position;
        Vector3 nestSize = nest.transform.localScale;

        return Mathf.Abs(position.x - nestCenter.x)/10 <= nestSize.x / 2f &&
               Mathf.Abs(position.z - nestCenter.z)/10 <= nestSize.z / 2f;
    }

    bool IsInFoodSource(Vector3 position)
    {
        foreach (var foodSource in foodSources)
        {
            Vector3 foodCenter = foodSource.transform.position;
            Vector3 foodSize = foodSource.transform.localScale;

            if (Mathf.Abs(position.x - foodCenter.x) <= foodSize.x / 2f &&
                Mathf.Abs(position.z - foodCenter.z) <= foodSize.z / 2f)
            {
                return true;
            }
        }
        return false;
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
