using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class ToyEpuck : Agent
{
    Rigidbody rBody;
    public float speed = 5f; // Speed of the robot's movement
    public Transform Target;

    void Start()
    {
        rBody = GetComponent<Rigidbody>();
        rBody.constraints = RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ;
    }

    public override void OnEpisodeBegin()
    {
        // Reset robot if it falls
        if (this.transform.localPosition.y < 0)
        {
            this.rBody.angularVelocity = Vector3.zero;
            this.rBody.linearVelocity = Vector3.zero;
            this.transform.localPosition = new Vector3(0, 0.01f, 0);
            this.transform.localRotation = Quaternion.identity;
        }

        // Move the target to a new spot
        Target.localPosition = new Vector3(Random.value * 8 - 4, 0.01f, Random.value * 8 - 4);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Target and Agent positions
        sensor.AddObservation(Target.localPosition - this.transform.localPosition);

        // Agent's orientation
        sensor.AddObservation(this.transform.forward);

        // Agent's velocities
        sensor.AddObservation(rBody.linearVelocity.x);
        sensor.AddObservation(rBody.linearVelocity.z);
        sensor.AddObservation(rBody.angularVelocity.y);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // Get the discrete action index
        int action = actionBuffers.DiscreteActions[0];

        // Move the robot based on the action
        switch (action)
        {
            case 0: // Forward
                rBody.MovePosition(transform.position + transform.forward * speed * Time.deltaTime);
                break;

            case 1: // Backward
                rBody.MovePosition(transform.position - transform.forward * speed * Time.deltaTime);
                break;

            case 2: // Turn Left
                transform.Rotate(Vector3.up, -90f * Time.deltaTime);
                break;

            case 3: // Turn Right
                transform.Rotate(Vector3.up, 90f * Time.deltaTime);
                break;

            case 4: // Stop
                break;
        }

        // Rewards
        float distanceToTarget = Vector3.Distance(this.transform.localPosition, Target.localPosition);

        // Accumulate reward if within the target area
        if (distanceToTarget < Target.localScale.x/2)
        {
            SetReward(0.01f); // Small reward for staying within the target
        }

        // Penalize for falling off
        if (this.transform.localPosition.y < 0)
        {
            AddReward(-1.0f); // Negative reward for falling
            EndEpisode();
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;

        if (Input.GetKey(KeyCode.W)) // Forward
            discreteActionsOut[0] = 0;
        else if (Input.GetKey(KeyCode.S)) // Backward
            discreteActionsOut[0] = 1;
        else if (Input.GetKey(KeyCode.A)) // Turn Left
            discreteActionsOut[0] = 2;
        else if (Input.GetKey(KeyCode.D)) // Turn Right
            discreteActionsOut[0] = 3;
        else
            discreteActionsOut[0] = -1; // No action
    }
}
