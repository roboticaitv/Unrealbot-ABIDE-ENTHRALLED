#pragma once
#include <math.h>

struct WheelTargets {
  float fl, fr, rl;
};

// Define your specific wheel angles here (in radians)
// Standard 120-degree symmetric setup, with X pointing forward:
const float ANGLE_FL = 150.0f * (M_PI / 180.0f); // Front-Left
const float ANGLE_FR = 30.0f  * (M_PI / 180.0f); // Front-Right
const float ANGLE_RL = 270.0f * (M_PI / 180.0f); // Rear-Left (Back)

const float ROBOT_RADIUS = 0.15f; // Distance from center to wheel (meters)

static inline float normalize_angle(float angle) {
  while (angle > M_PI) angle -= 2.0f * M_PI;
  while (angle <= -M_PI) angle += 2.0f * M_PI;
  return angle;
}

// 1. INVERSE KINEMATICS (Robot target -> Wheel target)
// vx, vy in m/s, omega in rad/s
static inline WheelTargets omni_ik(float vx, float vy, float omega) {
  float v_fl = -vx * sin(ANGLE_FL) + vy * cos(ANGLE_FL) + ROBOT_RADIUS * omega;
  float v_fr = -vx * sin(ANGLE_FR) + vy * cos(ANGLE_FR) + ROBOT_RADIUS * omega;
  float v_rl = -vx * sin(ANGLE_RL) + vy * cos(ANGLE_RL) + ROBOT_RADIUS * omega;
  
  return { v_fl, v_fr, v_rl };
}

// 2. PATH PLANNING (World Target -> Robot Vector)
// Calculates vx, vy, omega required to reach a specific X, Y, Theta
static inline void calculate_position_error(float x_curr, float y_curr, float theta_curr,
                              float x_goal, float y_goal, float theta_goal,
                              float &out_vx, float &out_vy, float &out_omega) {
  // Proportional gains for position control
  const float Kp_xy = 2.0f;
  const float Kp_th = 1.5f;

  float dx_world = x_goal - x_curr;
  float dy_world = y_goal - y_curr;
  float dtheta = normalize_angle(theta_goal - theta_curr);

  // Rotate world error into robot's local frame
  float cos_th = cosf(theta_curr);
  float sin_th = sinf(theta_curr);
  
  out_vx = (cos_th * dx_world + sin_th * dy_world) * Kp_xy;
  out_vy = (-sin_th * dx_world + cos_th * dy_world) * Kp_xy;
  out_omega = dtheta * Kp_th;
}

// 3. FORWARD KINEMATICS (Odometry)
// Converts delta wheel distances into World X, Y, Theta
static inline void updateOdometry(float d_fl, float d_fr, float d_rl, 
                    float &current_x, float &current_y, float &current_th) {
  
  // Matrix inverse for standard 120-degree setup
  float robot_dx = (-d_fl - d_fr + 2.0f * d_rl) / 3.0f; 
  float robot_dy = (-sqrt(3.0f) * d_fl + sqrt(3.0f) * d_fr) / 3.0f;
  float robot_dth = (d_fl + d_fr + d_rl) / (3.0f * ROBOT_RADIUS);

  float cos_th = cosf(current_th);
  float sin_th = sinf(current_th);

  current_x += robot_dx * cos_th - robot_dy * sin_th;
  current_y += robot_dx * sin_th + robot_dy * cos_th;
  current_th = normalize_angle(current_th + robot_dth);
}
