/*
 * Unity-ROS Bridge Example
 * Demonstrates basic communication between Unity and ROS 2
 * This script shows how to send and receive messages between Unity and ROS
 */

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using ROS2;

public class UnityROSBridge : MonoBehaviour
{
    // ROS2 components
    private ROS2UnityComponent ros2Unity;
    private UDPBase udpBase;

    // Publishers and Subscribers
    private ROS2Publisher<std_msgs.msg.Float64MultiArray> jointPublisher;
    private ROS2Subscriber<sensor_msgs.msg.JointState> jointSubscriber;

    // Robot joint control
    public Transform robotBase;
    public Transform joint1;
    public Transform joint2;
    public Transform joint3;

    // Joint angle limits
    private float[] jointLimitsMin = {-1.57f, -1.57f, -1.57f};
    private float[] jointLimitsMax = {1.57f, 1.57f, 1.57f};

    // Joint positions (in radians)
    private float[] currentJointPositions = {0.0f, 0.0f, 0.0f};
    private float[] targetJointPositions = {0.0f, 0.0f, 0.0f};

    // Speed control
    public float jointSpeed = 0.5f;

    void Start()
    {
        // Initialize ROS2 components
        ros2Unity = GetComponent<ROS2UnityComponent>();
        ros2Unity.Init();

        // Create publishers and subscribers
        jointPublisher = ros2Unity.node.CreatePublisher<std_msgs.msg.Float64MultiArray>("/joint_commands");
        jointSubscriber = ros2Unity.node.CreateSubscription<sensor_msgs.msg.JointState>("/joint_states", JointStateCallback);

        // Log initialization
        Debug.Log("Unity-ROS Bridge initialized");
    }

    void Update()
    {
        // Update joint positions based on target positions
        for (int i = 0; i < currentJointPositions.Length; i++)
        {
            currentJointPositions[i] = Mathf.MoveTowards(currentJointPositions[i],
                                                      targetJointPositions[i],
                                                      jointSpeed * Time.deltaTime);
        }

        // Apply joint rotations to Unity transforms
        if (joint1 != null) joint1.localRotation = Quaternion.Euler(0, 0, Mathf.Rad2Deg * currentJointPositions[0]);
        if (joint2 != null) joint2.localRotation = Quaternion.Euler(0, 0, Mathf.Rad2Deg * currentJointPositions[1]);
        if (joint3 != null) joint3.localRotation = Quaternion.Euler(0, 0, Mathf.Rad2Deg * currentJointPositions[2]);

        // Publish current joint states periodically
        if (Time.time % 0.1f < Time.deltaTime) // Publish at 10Hz
        {
            PublishJointStates();
        }
    }

    void JointStateCallback(sensor_msgs.msg.JointState msg)
    {
        // Update target joint positions from ROS messages
        if (msg.position.Count >= 3)
        {
            for (int i = 0; i < 3; i++)
            {
                // Apply joint limits
                targetJointPositions[i] = Mathf.Clamp((float)msg.position[i],
                                                    jointLimitsMin[i],
                                                    jointLimitsMax[i]);
            }
        }
    }

    void PublishJointStates()
    {
        // Create and publish joint state message
        var jointStateMsg = new sensor_msgs.msg.JointState();
        jointStateMsg.name = new List<string> { "joint1", "joint2", "joint3" };
        jointStateMsg.position = new List<double> {
            currentJointPositions[0],
            currentJointPositions[1],
            currentJointPositions[2]
        };
        jointStateMsg.header.stamp = ros2Unity.node.GetClock().Now().ToTimeMsg();
        jointStateMsg.header.frame_id = "base_link";

        // Publish the message
        jointSubscriber.node.GetPublisher<sensor_msgs.msg.JointState>("/unity_joint_states").Publish(jointStateMsg);
    }

    // Method to send joint commands to ROS
    public void SendJointCommands(float j1, float j2, float j3)
    {
        // Clamp values to limits
        j1 = Mathf.Clamp(j1, jointLimitsMin[0], jointLimitsMax[0]);
        j2 = Mathf.Clamp(j2, jointLimitsMin[1], jointLimitsMax[1]);
        j3 = Mathf.Clamp(j3, jointLimitsMin[2], jointLimitsMax[2]);

        // Set target positions
        targetJointPositions[0] = j1;
        targetJointPositions[1] = j2;
        targetJointPositions[2] = j3;

        // Optionally publish to ROS as well
        var cmdMsg = new std_msgs.msg.Float64MultiArray();
        cmdMsg.data = new List<double> { j1, j2, j3 };

        if (jointPublisher != null)
        {
            jointPublisher.Publish(cmdMsg);
        }
    }

    // Example method to move robot in a pattern
    public void MoveInPattern()
    {
        StartCoroutine(JointPattern());
    }

    IEnumerator JointPattern()
    {
        // Example pattern: move joints in sequence
        float[] pattern = { 0.5f, -0.5f, 0.0f, 0.5f, 0.5f, -0.5f, -0.5f, 0.0f, 0.0f };

        for (int i = 0; i < pattern.Length; i += 3)
        {
            if (i + 2 < pattern.Length)
            {
                SendJointCommands(pattern[i], pattern[i + 1], pattern[i + 2]);
                yield return new WaitForSeconds(2.0f);
            }
        }
    }

    void OnDestroy()
    {
        // Clean up ROS connections
        if (ros2Unity != null)
        {
            ros2Unity.Shutdown();
        }
    }
}