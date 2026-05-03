/*
 * ROBOT 6DOF - ARDUINO MEGA 2560 - STANDALONE MODE
 * 
 * Chức năng:
 * - Động học ngược (Inverse Kinematics)
 * - Nội suy quỹ đạo quintic mượt
 * - Chạy tự động không cần ROS
 * 
 * Tham số robot (từ MATLAB IK):
 * d1 = 0.2315m, a2 = 0.22168m, d3 = 0.22475m, d6 = 0.20264m
 */

#include <AccelStepper.h>

// ============================================================
// KHAI BÁO CHÂN ĐỘNG CƠ
// ============================================================
#define X_STEP_PIN 54
#define X_DIR_PIN 55
#define X_ENABLE_PIN 38


#define Z_STEP_PIN 46
#define Z_DIR_PIN 48
#define Z_ENABLE_PIN 62

#define Y_STEP_PIN 60
#define Y_DIR_PIN 61
#define Y_ENABLE_PIN 56

#define E0_STEP_PIN 26
#define E0_DIR_PIN 28
#define E0_ENABLE_PIN 24

#define E1_STEP_PIN 36
#define E1_DIR_PIN 34
#define E1_ENABLE_PIN 30

#define E2_STEP_PIN 47
#define E2_DIR_PIN 45
#define E2_ENABLE_PIN 32

#define GRIPPER_PIN 3  // Chân D3 điều khiển gripper (1=HIGH đóng, 0=LOW mở)

// ============================================================
// TẠO STEPPER
// ============================================================
AccelStepper joint1(AccelStepper::DRIVER, X_STEP_PIN, X_DIR_PIN);
AccelStepper joint2(AccelStepper::DRIVER, Z_STEP_PIN, Z_DIR_PIN);
AccelStepper joint3(AccelStepper::DRIVER, Y_STEP_PIN, Y_DIR_PIN);
AccelStepper joint4(AccelStepper::DRIVER, E0_STEP_PIN, E0_DIR_PIN);
AccelStepper joint5(AccelStepper::DRIVER, E1_STEP_PIN, E1_DIR_PIN);
AccelStepper joint6(AccelStepper::DRIVER, E2_STEP_PIN, E2_DIR_PIN);
AccelStepper* const JOINTS[6] = {&joint1, &joint2, &joint3, &joint4, &joint5, &joint6};

// ============================================================
// THÔNG SỐ ROBOT
// ============================================================
const float d1 = 0.2315;   // m
const float a2 = 0.22168;  // m
const float d3 = 0.22475;  // m
const float d6 = 0.20264;  // m

// Steps per revolution cho mỗi khớp
const long STEPS_PER_REV[6] = {27333, 18000, 18000, 6400, 12000, 8192};
// Home position offset (rad) - Robot's physical home position
// J4: offset = 0 vì bù -180° được trừ TRỰC TIẾP vào q_rad[3] sau khi IK, trước khi tính xung
// J5: offset = 0 vì bù -90° được trừ TRỰC TIẾP vào q_rad[4] sau khi IK, trước khi tính xung
const float HOME_OFFSET[6] = {0.0, PI/2, 0.0, 0.0, 0.0, -PI};

// Góc DH của các khớp tại vị trí HOME vật lý (tất cả motor ở step 0)
// J1=0, J2=PI/2 (tay thẳng đứng), J3=0
// J4=PI  (do loop trừ PI trước rad_to_steps)
// J5=PI/2 (do loop trừ PI/2 trước rad_to_steps)
// J6=-PI (HOME_OFFSET[5]=-PI)
const float HOME_Q_IK[6] = {0.0, PI/2, 0.0, PI, PI/2, -PI};

// Giới hạn góc (rad) - ĐO TƯƠNG ĐỐI SO VỚI HOME_Q_IK (khung motor)
// Ví dụ: Q_MAX[1]=PI/2 nghĩa là J2 được đi ±PI/2 từ vị trí home
const float Q_MIN[6] = {-PI, -PI/2, -2*PI/3, -PI, -5*PI/9, -PI};
const float Q_MAX[6] = {PI, PI/2, 2*PI/3, PI, 5*PI/9, PI};

// Góc khớp trước đó (để unwrap liên tục)
// Khởi tạo = HOME_Q_IK (robot bắt đầu ở home)
float q_prev[6] = {0.0, PI/2, 0.0, PI, PI/2, -PI};

// Góc IK nghiệm cuối cùng thành công (frame DH, trước bù J4/J5).
// Đây là tham chiếu THỰC TẾ để unwrap_angle tìm nghiệm gần vị trí hiện tại nhất.
// Khởi tạo = HOME_Q_IK
float q_current_ik[6] = {0.0, PI/2, 0.0, PI, PI/2, -PI};

// ============================================================
// CHẾ ĐỘ ĐIỀU KHIỂN
// ============================================================
enum ControlMode {
  MODE_QUEUE,   // Run queued G/GR commands
  MODE_MANUAL,  // Manual single target
  MODE_IDLE     // Stop
};

ControlMode control_mode = MODE_IDLE;

// Target position for manual mode
float target_p[3] = {0, 0, 0.855};
float target_rpy[3] = {0, 0, 0};
bool has_new_target = false;

// Queue of targets from G/GR commands
#define MAX_QUEUE 100
#define TASK_LIST_COUNT 3
#define MAX_TASK_ITEMS 20
#define ARC_STEPS 35
float queue_p[MAX_QUEUE][3];
float queue_rpy[MAX_QUEUE][3];
float queue_delay_s[MAX_QUEUE];
float queue_j6_extra[MAX_QUEUE]; // Extra rotation for Joint 6 (in radians)
bool queue_auto_orient[MAX_QUEUE]; // True if IK0 mode (auto find orientation)
int8_t queue_d3_state[MAX_QUEUE]; // Trạng thái chân D3: 1=HIGH, 0=LOW, -1=không thay đổi
float queue_gripper_delay_s[MAX_QUEUE]; // Thời gian chờ trước khi kích gripper (giây)
int queue_count = 0;
int queue_index = 0;
bool queue_running = false;
bool queue_step_active = false;
float last_g_center[3] = {0.0f, 0.0f, 0.0f};
bool last_g_center_valid = false;

const uint8_t TASK_TRIGGER_PINS[TASK_LIST_COUNT] = {2, 14, 15};
const char* const TASK_TRIGGER_NAMES[TASK_LIST_COUNT] = {"D2", "D14", "D15"};
const unsigned long TASK_TRIGGER_DEBOUNCE_MS = 40;
int16_t task_p_mm[TASK_LIST_COUNT][MAX_TASK_ITEMS][3];
int16_t task_rpy_deg10[TASK_LIST_COUNT][MAX_TASK_ITEMS][3];
uint16_t task_delay_cs[TASK_LIST_COUNT][MAX_TASK_ITEMS];
int16_t task_j6_extra_deg10[TASK_LIST_COUNT][MAX_TASK_ITEMS];
int8_t task_d3_state[TASK_LIST_COUNT][MAX_TASK_ITEMS];
uint16_t task_gripper_delay_cs[TASK_LIST_COUNT][MAX_TASK_ITEMS];
uint8_t task_count[TASK_LIST_COUNT] = {0, 0, 0};
bool task_trigger_prev_state[TASK_LIST_COUNT] = {false, false, false};
bool task_trigger_raw_state[TASK_LIST_COUNT] = {false, false, false};
bool task_trigger_latched[TASK_LIST_COUNT] = {false, false, false};
unsigned long task_trigger_last_change_ms[TASK_LIST_COUNT] = {0, 0, 0};

// Biến cho non-blocking movement
bool is_moving = false;
unsigned long move_delay_start = 0;
bool waiting_for_delay = false;
bool waiting_for_gripper = false;    // Đang chờ trước khi kích gripper
unsigned long gripper_delay_start = 0;

// Speed control (percentage: 50% = 0.5, 100% = 1.0, 200% = 2.0)
float speed_multiplier = 1.0;
const float SPEED_MIN = 0.1;  // 10% minimum
const float SPEED_MAX = 3.0;  // 300% maximum
// Nang tran toc do toan bo cac khop de test xem rung/on co den tu co khi hay khong.
const float BASE_MAX_SPEEDS[6] = {1550.0, 850.0, 3200.0, 650.0, 1200.0, 700.0};
const float BASE_MAX_ACCELS[6] = {1000.0, 2000.0, 3500.0, 1000.0, 1000.0, 1000.0};
const float COORD_MIN_SPEEDS[6] = {180.0, 120.0, 200.0, 100.0, 140.0, 100.0};
const float COORD_MIN_ACCELS[6] = {90.0, 60.0, 110.0, 40.0, 80.0, 50.0};
const unsigned int MIN_STEP_PULSE_US = 8;
const long SHORT_MOVE_THRESHOLD_STEPS = 900;
const float SHORT_MOVE_SPEED_SCALE_MIN = 0.42f;
const float SHORT_MOVE_ACCEL_SCALE_MIN = 0.25f;
const float SHORT_MOVE_FLOOR_SCALE_MIN = 0.50f;

// Gearbox tuning rieng cho J3: tang toc/ham nhanh hon de thoat som vung cong huong,
// nhung van giu 1 muc giam nhe khi doi chieu de tranh dap backlash cua hop so.
const float JOINT_SPEED_SCALES[6] = {1.00, 1.00, 0.85, 1.00, 1.00, 1.00};
const float JOINT_ACCEL_SCALES[6] = {1.00, 1.00, 0.55, 1.00, 1.00, 1.00};
const float JOINT_MIN_SPEED_SCALES[6] = {1.00, 1.00, 1.25, 1.00, 1.00, 1.00};
const float JOINT_MIN_ACCEL_SCALES[6] = {1.00, 1.00, 1.00, 1.00, 1.00, 1.00};
const float J3_REVERSE_SPEED_SCALE = 1.92;
const float J3_REVERSE_ACCEL_SCALE = 1.75;

// Doi 1 bien nay de debug profile chuyen dong.
enum MotionProfileMode {
  PROFILE_SMOOTH = 0,
  PROFILE_BALANCED = 1,
  PROFILE_FAST = 2
};

struct MotionProfile {
  const char* name;
  float speed_scale;
  float accel_scale;
};

uint8_t motion_profile_mode = PROFILE_BALANCED;

const MotionProfile MOTION_PROFILES[] = {
  {"SMOOTH",   1.5, 1.20},
  {"BALANCED", 1.6, 1.30},
  {"FAST",     1.5, 1.4}
};

long active_move_targets[6] = {0, 0, 0, 0, 0, 0};
long active_move_delta[6] = {0, 0, 0, 0, 0, 0};
long previous_move_delta[6] = {0, 0, 0, 0, 0, 0};
float active_move_speeds[6] = {0, 0, 0, 0, 0, 0};
float active_move_accels[6] = {0, 0, 0, 0, 0, 0};
long active_lead_distance = 0;
bool coordinated_plan_active = false;

const MotionProfile& currentMotionProfile();
void updateMotorSpeeds();
void clearCoordinatedMotionPlan();
void refreshActiveMoveSpeeds();
void beginCoordinatedMove(const long* targetSteps);
bool enqueueTaskCommand(int task_idx,
                        float x, float y, float z,
                        float r, float p, float yaw,
                        float delay_s, float j6_extra_deg,
                        int8_t d3_state, float gripper_delay_s);
bool startTaskList(int task_idx, const char* source_name, bool quiet_failures);
void clearTaskList(int task_idx);
void printTaskList(int task_idx);
void checkTaskTriggerInputs();

// ============================================================
// WAYPOINTS (từ MATLAB)
// ============================================================
const int NUM_WAYPOINTS = 4;
float P_waypoints[4][3] = {
  {0.0,  0.00,  0.88057},  // P0
  {0.4,  0.30,  0.2315},    // P1
  {0.2,  0.10,  0.60},    // P2
  {0.4,  0.10,  0.60}     // P3
};

float R_waypoints[4][3] = {
  {0.0,   0.0,     0.0},    // R0 [roll, pitch, yaw]
  {0.0,   1.5708,  0.0},    // R1
  {0.0,   1.5708,  0.0},    // R2
  {0.0,   1.5708,  0.0}     // R3
};

// RPY unwrapped (tính một lần trong setup)
float Ru_waypoints[4][3];

const float T_TOTAL = 35; // giây
const int NUM_SEGMENTS = 3;
const float T_SEG = T_TOTAL / NUM_SEGMENTS;

// ============================================================
// CẤU TRÚC DỮ LIỆU
// ============================================================
struct Vec3 {
  float x, y, z;
};

struct Mat3 {
  float m[3][3];
};

struct DH_Transform {
  float m[4][4];
};

// ============================================================
// HÀM TOÁN HỌC CƠ BẢN
// ============================================================

// Unwrap góc về gần ref nhất
float unwrap_angle(float angle, float ref) {
  float da = angle - ref;
  da = fmod(da + PI, 2*PI) - PI;
  return ref + da;
}

// Unwrap vector 3D
void unwrap_vec3(float* result, const float* vec, const float* ref) {
  for (int i = 0; i < 3; i++) {
    result[i] = unwrap_angle(vec[i], ref[i]);
  }
}

// RPY (ZYX) to Rotation Matrix
void rpy2rotm_ZYX(Mat3* R, float roll, float pitch, float yaw) {
  float cr = cos(roll),  sr = sin(roll);
  float cp = cos(pitch), sp = sin(pitch);
  float cy = cos(yaw),   sy = sin(yaw);
  
  R->m[0][0] = cy*cp;
  R->m[0][1] = cy*sp*sr - sy*cr;
  R->m[0][2] = cy*sp*cr + sy*sr;
  
  R->m[1][0] = sy*cp;
  R->m[1][1] = sy*sp*sr + cy*cr;
  R->m[1][2] = sy*sp*cr - cy*sr;
  
  R->m[2][0] = -sp;
  R->m[2][1] = cp*sr;
  R->m[2][2] = cp*cr;
}

// Standard DH transform
void dh_std(DH_Transform* T, float a, float alpha, float d, float theta) {
  float ct = cos(theta), st = sin(theta);
  float ca = cos(alpha), sa = sin(alpha);
  
  T->m[0][0] = ct;      T->m[0][1] = -st*ca;   T->m[0][2] = st*sa;    T->m[0][3] = a*ct;
  T->m[1][0] = st;      T->m[1][1] = ct*ca;    T->m[1][2] = -ct*sa;   T->m[1][3] = a*st;
  T->m[2][0] = 0;       T->m[2][1] = sa;       T->m[2][2] = ca;       T->m[2][3] = d;
  T->m[3][0] = 0;       T->m[3][1] = 0;        T->m[3][2] = 0;        T->m[3][3] = 1;
}

// Nhân 2 ma trận 4x4
void mat4_mult(DH_Transform* result, const DH_Transform* A, const DH_Transform* B) {
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      result->m[i][j] = 0;
      for (int k = 0; k < 4; k++) {
        result->m[i][j] += A->m[i][k] * B->m[k][j];
      }
    }
  }
}

// Nhân 2 ma trận 3x3
void mat3_mult(Mat3* result, const Mat3* A, const Mat3* B) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      result->m[i][j] = 0;
      for (int k = 0; k < 3; k++) {
        result->m[i][j] += A->m[i][k] * B->m[k][j];
      }
    }
  }
}

// Chuyển vị ma trận 3x3
void mat3_transpose(Mat3* result, const Mat3* A) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      result->m[i][j] = A->m[j][i];
    }
  }
}

// ============================================================
// WRIST IK (Spherical wrist)
// ============================================================
bool wrist_from_R36(const Mat3* R36, float* q4, float* q5, float* q6) {
  float s5 = sqrt(R36->m[0][2]*R36->m[0][2] + R36->m[1][2]*R36->m[1][2]);
  float c5 = R36->m[2][2];
  *q5 = atan2(s5, c5);
  
  if (s5 < 1e-6) {
    // Singularity
    *q4 = 0;
    *q6 = atan2(R36->m[1][0], R36->m[0][0]);
    return true;
  }
  
  *q4 = atan2(R36->m[1][2], R36->m[0][2]);
  *q6 = atan2(-R36->m[2][1], R36->m[2][0]);
  return true;
}

// Mức IK đã dùng (1=J1235, 2=J12345, 3=J123456)
int g_ik_level = 3;
float g_ik_motion_cost = 1e30f;

// Ngưỡng sai số FK để chấp nhận nghiệm rút gọn
const float IK_POS_TOL = 0.001f;   // 1 mm
const float IK_ROT_TOL = 0.15f;    // ~6° (Frobenius norm)
const float IK_COST_TIE_EPS = 1e-5f;

// ============================================================
// KIỂM TRA WRIST BẰNG CÔNG THỨC PHÂN TÍCH (không cần full FK)
// → nhanh hơn ~10× so với forward_kinematics_rot trong vòng lặp IK
// ============================================================

// R36 phân tích cho (q4,q5,q6) theo DH robot này:
// A34: alpha=-π/2  A45: alpha=+π/2  A56: alpha=0
// R36 = [ c4c5c6-s4s6,  -c4c5s6-s4c6,  c4s5 ]
//        [ s4c5c6+c4s6,  -s4c5s6+c4c6,  s4s5 ]
//        [ -s5c6,         s5s6,           c5  ]
// Sai số góc: Frobenius(R36_achieved - R36_target)
float wrist_rot_error(const Mat3* R36_tgt, float q4, float q5, float q6) {
  float c4=cos(q4), s4=sin(q4);
  float c5=cos(q5), s5=sin(q5);
  float c6=cos(q6), s6=sin(q6);
  float R[3][3] = {
    {c4*c5*c6-s4*s6,  -c4*c5*s6-s4*c6,  c4*s5},
    {s4*c5*c6+c4*s6,  -s4*c5*s6+c4*c6,  s4*s5},
    {-s5*c6,           s5*s6,            c5   }
  };
  float err = 0;
  for (int ii=0; ii<3; ii++)
    for (int jj=0; jj<3; jj++) { float d=R[ii][jj]-R36_tgt->m[ii][jj]; err+=d*d; }
  return sqrt(err);
}

// Sai số vị trí end-effector:
// p_ee = p_B + d6 * R03 * R36_col2,  với R36_col2 = [c4*s5; s4*s5; c5]
float wrist_pos_error(const Vec3* p_tgt, const Vec3* pB,
                      const Mat3* R03, float q4, float q5) {
  float c4=cos(q4), s4=sin(q4), c5=cos(q5), s5=sin(q5);
  float z6[3] = {c4*s5, s4*s5, c5};   // R36 column 2
  float z6w[3] = {0,0,0};
  for (int ii=0; ii<3; ii++)
    for (int jj=0; jj<3; jj++)
      z6w[ii] += R03->m[ii][jj] * z6[jj]; // R03 * z6
  float dx = pB->x + d6*z6w[0] - p_tgt->x;
  float dy = pB->y + d6*z6w[1] - p_tgt->y;
  float dz = pB->z + d6*z6w[2] - p_tgt->z;
  return sqrt(dx*dx + dy*dy + dz*dz);
}

float wrap_pm_pi(float angle) {
  while (angle > PI) angle -= 2*PI;
  while (angle < -PI) angle += 2*PI;
  return angle;
}

void build_continuous_candidate(const float* q_raw, float* q_continuous) {
  for (int k = 0; k < 6; k++) {
    q_continuous[k] = unwrap_angle(q_raw[k], q_prev[k]);
  }
}

bool candidate_within_limits(const float* q_candidate) {
  for (int k = 0; k < 6; k++) {
    float dq = wrap_pm_pi(q_candidate[k] - HOME_Q_IK[k]);
    if (dq < Q_MIN[k] || dq > Q_MAX[k]) {
      return false;
    }
  }
  return true;
}

float candidate_motion_cost(const float* q_candidate) {
  float cost = 0.0f;
  for (int k = 0; k < 6; k++) {
    float dq = q_candidate[k] - q_prev[k];
    cost += dq * dq;
  }
  return cost;
}

bool is_better_candidate(float motion_cost, int level_idx,
                         bool best_found, float best_motion_cost, int best_level_idx) {
  if (!best_found) return true;
  if (motion_cost < best_motion_cost - IK_COST_TIE_EPS) return true;
  if (fabsf(motion_cost - best_motion_cost) <= IK_COST_TIE_EPS &&
      level_idx < best_level_idx) {
    return true;
  }
  return false;
}

float steps_to_ik_angle(int joint_idx, long steps) {
  float turns = (2.0f * PI * (float)steps) / (float)STEPS_PER_REV[joint_idx];
  switch (joint_idx) {
    case 0: return HOME_OFFSET[0] + turns;
    case 1: return HOME_OFFSET[1] - turns;
    case 2: return HOME_OFFSET[2] + turns;
    case 3: return (-turns) + PI;
    case 4: return (-turns) + PI / 2.0f;
    case 5: return HOME_OFFSET[5] - turns;
    default: return 0.0f;
  }
}

void sync_ik_state_from_steppers(float* q_ik) {
  for (int i = 0; i < 6; i++) {
    q_ik[i] = steps_to_ik_angle(i, JOINTS[i]->currentPosition());
  }
}

// ============================================================
// INVERSE KINEMATICS - CORE
// Phân cấp 3 mức:
//   Level 1: J1,J2,J3,J5  (q4=0, q6=0) - ít khớp nhất
//   Level 2: J1,J2,J3,J4,J5 (q6=0)
//   Level 3: Đầy đủ 6 khớp (fallback)
// ============================================================
bool ik_6dof_core(const Mat3* R06, const Vec3* p06, float* q_result) {
  // Wrist center B
  Vec3 pB;
  pB.x = p06->x - d6 * R06->m[0][2];
  pB.y = p06->y - d6 * R06->m[1][2];
  pB.z = p06->z - d6 * R06->m[2][2];

  float x = pB.x, y = pB.y, z = pB.z;

  // q1
  float q1 = atan2(y, x);

  // q2, q3 (geometry)
  float r = sqrt(x*x + y*y);
  float s = z - d1;
  float D2 = r*r + s*s;
  float c3 = (D2 - a2*a2 - d3*d3) / (2.0*a2*d3);

  if (c3 < -1.0f - 1e-6f || c3 > 1.0f + 1e-6f) {
    g_ik_motion_cost = 1e30f;
    return false;
  }
  c3 = constrain(c3, -1.0, 1.0);
  float s3 = sqrt(max(0.0f, 1.0f - c3*c3));
  float q3_list[2] = {atan2(s3, c3), atan2(-s3, c3)};

  // Sắp xếp q3_list: nhánh gần q_prev[2] nhất lên trước
  // (tiebreaker khi cost bằng nhau — cả 2 nhánh đều được duyệt)
  {
    float d0 = fabsf(unwrap_angle(q3_list[0], q_prev[2]) - q_prev[2]);
    float d1 = fabsf(unwrap_angle(q3_list[1], q_prev[2]) - q_prev[2]);
    if (d1 < d0) {
      float tmp = q3_list[0]; q3_list[0] = q3_list[1]; q3_list[1] = tmp;
    }
  }

  // Lưu nghiệm tốt nhất cho từng mức (0=L1, 1=L2, 2=L3)
  // Duyệt TẤT CẢ 4 tổ hợp (2 q3 × 2 q5), cost function tự chọn nghiệm
  // có tổng dịch chuyển khớp nhỏ nhất → không cần break sớm
  float best_motion_cost = 1e30f;
  float best_q[6];
  int best_level_idx = 3;
  bool found_any = false;

  for (int i = 0; i < 2; i++) {
    float q3 = q3_list[i];
    float q2 = atan2(s, r) - atan2(d3*sin(q3), a2 + d3*cos(q3));

    // Tính R03
    DH_Transform A01, A12, A23, T01_12, T03;
    dh_std(&A01, 0,  PI/2, d1, q1);
    dh_std(&A12, a2, 0,    0,  q2);
    dh_std(&A23, 0,  PI/2, 0,  q3 + PI/2);
    mat4_mult(&T01_12, &A01, &A12);
    mat4_mult(&T03, &T01_12, &A23);

    Mat3 R03;
    for (int ii = 0; ii < 3; ii++)
      for (int jj = 0; jj < 3; jj++)
        R03.m[ii][jj] = T03.m[ii][jj];

    Mat3 R03_T, R36;
    mat3_transpose(&R03_T, &R03);
    mat3_mult(&R36, &R03_T, R06);

    // ── Hai nghiệm q5 (và q4, q6 tương ứng) ─────────────────
    // Nghiệm A: q5 > 0,  Nghiệm B: q5 < 0, q4 và q6 lệch π
    float s5_nat = sqrt(R36.m[0][2]*R36.m[0][2] + R36.m[1][2]*R36.m[1][2]);
    float c5_nat = R36.m[2][2];

    float q5_list[2], q4_list[2], q6_list[2];
    if (s5_nat < 1e-6f) {
      // Kỳ dị: chỉ 1 nghiệm, nhân đôi để logic bên dưới không đổi
      q5_list[0] = q5_list[1] = 0.0f;
      q4_list[0] = q4_list[1] = 0.0f;
      q6_list[0] = q6_list[1] = atan2(R36.m[1][0], R36.m[0][0]);
    } else {
      // Nghiệm A
      q5_list[0] = atan2( s5_nat, c5_nat);
      q4_list[0] = atan2( R36.m[1][2],  R36.m[0][2]);
      q6_list[0] = atan2(-R36.m[2][1],  R36.m[2][0]);
      // Nghiệm B (q5 đổi dấu → q4 đổi dấu phân tử, q6 đổi dấu phân tử)
      q5_list[1] = atan2(-s5_nat, c5_nat);
      q4_list[1] = atan2(-R36.m[1][2], -R36.m[0][2]);
      q6_list[1] = atan2( R36.m[2][1], -R36.m[2][0]);
    }

    // Sắp xếp q5_list: nghiệm gần q_prev[4] nhất lên trước
    // (tiebreaker khi cost bằng nhau — cả 2 nhánh đều được duyệt)
    {
      float d0 = fabsf(unwrap_angle(q5_list[0], q_prev[4]) - q_prev[4]);
      float d1 = fabsf(unwrap_angle(q5_list[1], q_prev[4]) - q_prev[4]);
      if (d1 < d0) {
        float tmp;
        tmp = q5_list[0]; q5_list[0] = q5_list[1]; q5_list[1] = tmp;
        tmp = q4_list[0]; q4_list[0] = q4_list[1]; q4_list[1] = tmp;
        tmp = q6_list[0]; q6_list[0] = q6_list[1]; q6_list[1] = tmp;
      }
    }

    for (int j = 0; j < 2; j++) {
      // Duyệt cả 2 nhánh q5, cost function tự chọn nghiệm tối ưu

      float q5 = q5_list[j];
      float q4 = q4_list[j];
      float q6 = q6_list[j];

      // ── Level 1: q4=0, q5, q6=0 ────────────────────────────
      // Kiểm tra giới hạn trong KHUNG MOTOR (tương đối so với HOME_Q_IK)
      // → Q_MIN/Q_MAX là ±giới hạn từ vị trí home, NOT từ DH zero
      // Chỉ dùng unwrap để tính cost di chuyển
      {
        float q_raw[6] = {q1, q2, q3, 0.0f, q5, 0.0f};
        float q_candidate[6];
        build_continuous_candidate(q_raw, q_candidate);
        if (candidate_within_limits(q_candidate)) {
          float pos_err = wrist_pos_error(p06, &pB, &R03, q_candidate[3], q_candidate[4]);
          float rot_err = wrist_rot_error(&R36, q_candidate[3], q_candidate[4], q_candidate[5]);
          if (pos_err < IK_POS_TOL && rot_err < IK_ROT_TOL) {
            float motion_cost = candidate_motion_cost(q_candidate);
            if (is_better_candidate(motion_cost, 0, found_any, best_motion_cost, best_level_idx)) {
              best_motion_cost = motion_cost;
              best_level_idx = 0;
              for (int k = 0; k < 6; k++) best_q[k] = q_candidate[k];
              found_any = true;
            }
          }
        }
      }

      // ── Level 2: q4, q5, q6=0 ──────────────────────────────
      {
        float q_raw[6] = {q1, q2, q3, q4, q5, 0.0f};
        float q_candidate[6];
        build_continuous_candidate(q_raw, q_candidate);
        if (candidate_within_limits(q_candidate)) {
          float pos_err = wrist_pos_error(p06, &pB, &R03, q_candidate[3], q_candidate[4]);
          float rot_err = wrist_rot_error(&R36, q_candidate[3], q_candidate[4], q_candidate[5]);
          if (pos_err < IK_POS_TOL && rot_err < IK_ROT_TOL) {
            float motion_cost = candidate_motion_cost(q_candidate);
            if (is_better_candidate(motion_cost, 1, found_any, best_motion_cost, best_level_idx)) {
              best_motion_cost = motion_cost;
              best_level_idx = 1;
              for (int k = 0; k < 6; k++) best_q[k] = q_candidate[k];
              found_any = true;
            }
          }
        }
      }

      // ── Level 3: q4, q5, q6 đầy đủ (fallback) ─────────────
      {
        float q_raw[6] = {q1, q2, q3, q4, q5, q6};
        float q_candidate[6];
        build_continuous_candidate(q_raw, q_candidate);
        if (candidate_within_limits(q_candidate)) {
          float motion_cost = candidate_motion_cost(q_candidate);
          if (is_better_candidate(motion_cost, 2, found_any, best_motion_cost, best_level_idx)) {
            best_motion_cost = motion_cost;
            best_level_idx = 2;
            for (int k = 0; k < 6; k++) best_q[k] = q_candidate[k];
            found_any = true;
          }
        }
      }
    } // end for j (q5 branches)
  } // end for i (q3 branches)

  // Chọn nghiệm theo ưu tiên: Level 1 > Level 2 > Level 3
  if (found_any) {
    for (int k = 0; k < 6; k++) q_result[k] = best_q[k];
    g_ik_level = best_level_idx + 1;
    g_ik_motion_cost = best_motion_cost;
    return true;
  }

  g_ik_motion_cost = 1e30f;
  return false;
}

// ============================================================
// INVERSE KINEMATICS - PUBLIC INTERFACE
// ============================================================
bool ik6dof_from_p06_rpy(float px, float py, float pz, 
                         float roll, float pitch, float yaw,
                         float* q_rad) {
  // Đặt q_prev = góc IK hiện tại để unwrap_angle tham chiếu đúng
  // → IK chọn nghiệm GẦN VỊ TRÍ HIỆN TẠI, tránh vượt giới hạn
  sync_ik_state_from_steppers(q_prev);
  for (int k = 0; k < 6; k++) q_current_ik[k] = q_prev[k];

  Mat3 R06;
  rpy2rotm_ZYX(&R06, roll, pitch, yaw);
  
  Vec3 p06 = {px, py, pz};
  
  bool valid = ik_6dof_core(&R06, &p06, q_rad);
  
  if (!valid) {
    // Giữ nguyên góc hiện tại khi thất bại
    for (int k = 0; k < 6; k++) {
      q_rad[k] = q_current_ik[k];
    }
  }
  // KHÔNG reset q_prev về 0 — q_current_ik được cập nhật ở loop sau khi move thành công
  return valid;
}

// ============================================================
// FORWARD KINEMATICS - Verify position
// ============================================================
void forward_kinematics(const float* q_rad, float* pos_xyz) {
  // Tính động học thuận để lấy vị trí end-effector từ góc khớp
  DH_Transform A01, A12, A23, A34, A45, A56;
  DH_Transform T01_12, T01_23, T01_34, T01_45, T06;
  
  // DH parameters
  dh_std(&A01, 0,  PI/2, d1, q_rad[0]);
  dh_std(&A12, a2, 0,    0,  q_rad[1]);
  dh_std(&A23, 0,  PI/2, 0,  q_rad[2] + PI/2);
  dh_std(&A34, 0, -PI/2, d3, q_rad[3]);
  dh_std(&A45, 0,  PI/2, 0,  q_rad[4]);
  dh_std(&A56, 0,  0,    d6, q_rad[5]);
  
  // Nhân dần các ma trận
  mat4_mult(&T01_12, &A01, &A12);
  mat4_mult(&T01_23, &T01_12, &A23);
  mat4_mult(&T01_34, &T01_23, &A34);
  mat4_mult(&T01_45, &T01_34, &A45);
  mat4_mult(&T06, &T01_45, &A56);
  
  // Lấy vị trí từ cột cuối
  pos_xyz[0] = T06.m[0][3];
  pos_xyz[1] = T06.m[1][3];
  pos_xyz[2] = T06.m[2][3];
}

// ============================================================
// FORWARD KINEMATICS - Trả về cả vị trí và ma trận quay
// ============================================================
void forward_kinematics_rot(const float* q_rad, float* pos_xyz, Mat3* R_out) {
  DH_Transform A01, A12, A23, A34, A45, A56;
  DH_Transform T01_12, T01_23, T01_34, T01_45, T06;

  dh_std(&A01, 0,  PI/2, d1, q_rad[0]);
  dh_std(&A12, a2, 0,    0,  q_rad[1]);
  dh_std(&A23, 0,  PI/2, 0,  q_rad[2] + PI/2);
  dh_std(&A34, 0, -PI/2, d3, q_rad[3]);
  dh_std(&A45, 0,  PI/2, 0,  q_rad[4]);
  dh_std(&A56, 0,  0,    d6, q_rad[5]);

  mat4_mult(&T01_12, &A01, &A12);
  mat4_mult(&T01_23, &T01_12, &A23);
  mat4_mult(&T01_34, &T01_23, &A34);
  mat4_mult(&T01_45, &T01_34, &A45);
  mat4_mult(&T06,    &T01_45, &A56);

  pos_xyz[0] = T06.m[0][3];
  pos_xyz[1] = T06.m[1][3];
  pos_xyz[2] = T06.m[2][3];

  if (R_out) {
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        R_out->m[i][j] = T06.m[i][j];
  }
}

// Frobenius norm của (R1 - R2) - đo sai số ma trận quay
// Giá trị ~ 0.123 ≈ 5°, ~ 0.247 ≈ 10°
float rotation_error_frob(const Mat3* R1, const Mat3* R2) {
  float err = 0;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      float d = R1->m[i][j] - R2->m[i][j];
      err += d * d;
    }
  return sqrt(err);
}

// ============================================================
// AUTO-FIND VALID ORIENTATION (for IK0 mode)
// ============================================================
bool ik6dof_auto_orient(float px, float py, float pz, 
                       float* q_rad, float* found_rpy) {
  // Thử nhiều hướng khác nhau để tìm ra hướng hợp lệ
  // Ưu tiên các hướng thông dụng: hướng xuống, hướng ngang, v.v.
  
  const int NUM_TRIES = 12; // Reduced from 24 to speed up
  float test_orientations[NUM_TRIES][3] = {
    // [roll, pitch, yaw] in radians - Most common orientations first
    {0.0, 0.0, 0.0},          // Down, no rotation
    {0.0, PI/2, 0.0},         // Forward horizontal
    {0.0, -PI/2, 0.0},        // Backward horizontal
    {0.0, 0.0, PI/2},         // Down, rotated 90°
    {0.0, 0.0, PI},           // Down, rotated 180°
    {0.0, 0.0, -PI/2},        // Down, rotated -90°
    {0.0, PI/4, 0.0},         // 45° forward
    {0.0, -PI/4, 0.0},        // 45° backward
    {0.0, PI/2, PI/2},        // Horizontal, 90° yaw
    {0.0, PI/2, -PI/2},       // Horizontal, -90° yaw
    {0.0, PI/3, 0.0},         // 60° tilt
    {0.0, 2*PI/3, 0.0}        // 120° tilt
  };
  
  // Thử từng hướng (giảm debug để tránh trệo)
  bool found_any = false;
  float best_motion_cost = 1e30f;
  float best_q[6];
  float best_rpy_local[3] = {0.0f, 0.0f, 0.0f};
  int best_level_idx = 3;
  int best_orientation_idx = NUM_TRIES;
  Vec3 p06 = {px, py, pz};
  sync_ik_state_from_steppers(q_current_ik);

  for (int i = 0; i < NUM_TRIES; i++) {
    float roll = test_orientations[i][0];
    float pitch = test_orientations[i][1];
    float yaw = test_orientations[i][2];
    
    // Tham chiếu từ góc hiện tại trước mỗi lần thử
    for (int k = 0; k < 6; k++) q_prev[k] = q_current_ik[k];

    Mat3 R06;
    rpy2rotm_ZYX(&R06, roll, pitch, yaw);
    float q_try[6];
    if (ik_6dof_core(&R06, &p06, q_try)) {
      // Tìm thấy hướng hợp lệ!
      int level_idx = g_ik_level - 1;
      bool better = is_better_candidate(g_ik_motion_cost, level_idx,
                                        found_any, best_motion_cost, best_level_idx);
      if (!better && found_any &&
          fabsf(g_ik_motion_cost - best_motion_cost) <= IK_COST_TIE_EPS &&
          level_idx == best_level_idx &&
          i < best_orientation_idx) {
        better = true;
      }

      if (better) {
        found_any = true;
        best_motion_cost = g_ik_motion_cost;
        best_level_idx = level_idx;
        best_orientation_idx = i;
        for (int k = 0; k < 6; k++) best_q[k] = q_try[k];
        best_rpy_local[0] = roll;
        best_rpy_local[1] = pitch;
        best_rpy_local[2] = yaw;
      }
    }
  }
  
  // Không tìm thấy hướng hợp lệ
  if (found_any) {
    for (int k = 0; k < 6; k++) q_rad[k] = best_q[k];
    found_rpy[0] = best_rpy_local[0];
    found_rpy[1] = best_rpy_local[1];
    found_rpy[2] = best_rpy_local[2];
    g_ik_level = best_level_idx + 1;
    g_ik_motion_cost = best_motion_cost;
    return true;
  }

  Serial.println(F("  ERROR: No valid orientation found!"));
  for (int k = 0; k < 6; k++) q_rad[k] = q_current_ik[k];
  return false;
}

// ============================================================
// QUINTIC POLYNOMIAL INTERPOLATION với velocity & acceleration
// ============================================================
void quintic_va(float t, float T, 
                const float* p0, const float* p1,
                const float* v0, const float* v1,
                const float* a0, const float* a1,
                float* result, int dim) {
  
  if (t <= 0) {
    for (int i = 0; i < dim; i++) result[i] = p0[i];
    return;
  }
  if (t >= T) {
    for (int i = 0; i < dim; i++) result[i] = p1[i];
    return;
  }
  
  float T2 = T*T, T3 = T2*T, T4 = T3*T, T5 = T4*T;
  
  for (int i = 0; i < dim; i++) {
    float c0 = p0[i];
    float c1 = v0[i];
    float c2 = 0.5 * a0[i];
    
    // Solve for c3, c4, c5
    float rhs1 = p1[i] - (c0 + c1*T + c2*T2);
    float rhs2 = v1[i] - (c1 + 2*c2*T);
    float rhs3 = a1[i] - (2*c2);
    
    // Matrix inverse (3x3)
    float M11=T3,   M12=T4,    M13=T5;
    float M21=3*T2, M22=4*T3,  M23=5*T4;
    float M31=6*T,  M32=12*T2, M33=20*T3;
    
    float detM = M11*(M22*M33 - M23*M32) - M12*(M21*M33 - M23*M31) + M13*(M21*M32 - M22*M31);
    
    float i11 = (M22*M33 - M23*M32) / detM;
    float i12 = (M13*M32 - M12*M33) / detM;
    float i13 = (M12*M23 - M13*M22) / detM;
    float i21 = (M23*M31 - M21*M33) / detM;
    float i22 = (M11*M33 - M13*M31) / detM;
    float i23 = (M13*M21 - M11*M23) / detM;
    float i31 = (M21*M32 - M22*M31) / detM;
    float i32 = (M12*M31 - M11*M32) / detM;
    float i33 = (M11*M22 - M12*M21) / detM;
    
    float c3 = i11*rhs1 + i12*rhs2 + i13*rhs3;
    float c4 = i21*rhs1 + i22*rhs2 + i23*rhs3;
    float c5 = i31*rhs1 + i32*rhs2 + i33*rhs3;
    
    float tt = t;
    float tt2 = tt*tt;
    float tt3 = tt2*tt;
    float tt4 = tt3*tt;
    float tt5 = tt4*tt;
    
    result[i] = c0 + c1*tt + c2*tt2 + c3*tt3 + c4*tt4 + c5*tt5;
  }
}

// ============================================================
// TRAJECTORY INTERPOLATION (Multi-segment smooth)
// ============================================================
void traj_interp_xyz_rpy_smooth(float t, float* p06, float* rpy) {
  // Wrap time
  if (t < 0) t = 0;
  t = fmod(t, T_TOTAL);
  
  int seg = (int)(t / T_SEG);
  if (seg >= NUM_SEGMENTS) seg = NUM_SEGMENTS - 1;
  
  float tau = t - seg * T_SEG;
  
  // Catmull-Rom velocities
  float vP[4][3], vR[4][3];
  float aP[4][3], aR[4][3]; // Zero acceleration
  
  // Waypoint 0
  for (int i = 0; i < 3; i++) {
    vP[0][i] = 0;
    vR[0][i] = 0;
    aP[0][i] = 0;
    aR[0][i] = 0;
  }
  
  // Waypoint 1
  for (int i = 0; i < 3; i++) {
    vP[1][i] = (P_waypoints[2][i] - P_waypoints[0][i]) / (2*T_SEG);
    vR[1][i] = (Ru_waypoints[2][i] - Ru_waypoints[0][i]) / (2*T_SEG);
    aP[1][i] = 0;
    aR[1][i] = 0;
  }
  
  // Waypoint 2
  for (int i = 0; i < 3; i++) {
    vP[2][i] = (P_waypoints[3][i] - P_waypoints[1][i]) / (2*T_SEG);
    vR[2][i] = (Ru_waypoints[3][i] - Ru_waypoints[1][i]) / (2*T_SEG);
    aP[2][i] = 0;
    aR[2][i] = 0;
  }
  
  // Waypoint 3
  for (int i = 0; i < 3; i++) {
    vP[3][i] = 0;
    vR[3][i] = 0;
    aP[3][i] = 0;
    aR[3][i] = 0;
  }
  
  // Interpolate based on segment
  if (seg == 0) {
    quintic_va(tau, T_SEG, P_waypoints[0], P_waypoints[1], vP[0], vP[1], aP[0], aP[1], p06, 3);
    quintic_va(tau, T_SEG, Ru_waypoints[0], Ru_waypoints[1], vR[0], vR[1], aR[0], aR[1], rpy, 3);
  } else if (seg == 1) {
    quintic_va(tau, T_SEG, P_waypoints[1], P_waypoints[2], vP[1], vP[2], aP[1], aP[2], p06, 3);
    quintic_va(tau, T_SEG, Ru_waypoints[1], Ru_waypoints[2], vR[1], vR[2], aR[1], aR[2], rpy, 3);
  } else {
    quintic_va(tau, T_SEG, P_waypoints[2], P_waypoints[3], vP[2], vP[3], aP[2], aP[3], p06, 3);
    quintic_va(tau, T_SEG, Ru_waypoints[2], Ru_waypoints[3], vR[2], vR[3], aR[2], aR[3], rpy, 3);
  }
}

// ============================================================
// RAD TO STEPS CONVERSION
// ============================================================
void rad_to_steps(const float* q_rad, long* steps) {
  // Convert absolute joint angles to steps, accounting for home position offset
  // Lưu ý: q_rad[4] đã được trừ 90° trước khi gọi hàm này (xem loop)
  steps[0] = (long)((q_rad[0] - HOME_OFFSET[0]) / (2*PI) * STEPS_PER_REV[0]);
  steps[1] = -(long)((q_rad[1] - HOME_OFFSET[1]) / (2*PI) * STEPS_PER_REV[1]); // Inverted
  steps[2] = (long)((q_rad[2] - HOME_OFFSET[2]) / (2*PI) * STEPS_PER_REV[2]);
  steps[3] = -(long)((q_rad[3] - HOME_OFFSET[3]) / (2*PI) * STEPS_PER_REV[3]); // Inverted
  steps[4] = -(long)((q_rad[4] - HOME_OFFSET[4]) / (2*PI) * STEPS_PER_REV[4]); // Inverted
  steps[5] = -(long)((q_rad[5] - HOME_OFFSET[5]) / (2*PI) * STEPS_PER_REV[5]); // Inverted
}

long nearest_equivalent_steps(long target_steps, long current_steps, long steps_per_rev) {
  long aligned_steps = target_steps;
  long delta = aligned_steps - current_steps;
  long half_rev = steps_per_rev / 2;

  while (delta > half_rev) {
    aligned_steps -= steps_per_rev;
    delta -= steps_per_rev;
  }
  while (delta < -half_rev) {
    aligned_steps += steps_per_rev;
    delta += steps_per_rev;
  }

  long best_steps = aligned_steps;
  long best_distance = labs(best_steps - current_steps);
  long alt_steps[2] = {aligned_steps - steps_per_rev, aligned_steps + steps_per_rev};
  for (int i = 0; i < 2; i++) {
    long alt_distance = labs(alt_steps[i] - current_steps);
    if (alt_distance < best_distance ||
        (alt_distance == best_distance && labs(alt_steps[i]) < labs(best_steps))) {
      best_steps = alt_steps[i];
      best_distance = alt_distance;
    }
  }

  return best_steps;
}

void align_periodic_joint_targets(long* steps) {
  const int periodic_joints[] = {3, 5}; // J4, J6
  for (int i = 0; i < 2; i++) {
    int joint_idx = periodic_joints[i];
    long current_steps = JOINTS[joint_idx]->currentPosition();
    steps[joint_idx] = nearest_equivalent_steps(steps[joint_idx],
                                                current_steps,
                                                STEPS_PER_REV[joint_idx]);
  }
}

// ============================================================
// UPDATE MOTOR SPEEDS
// ============================================================
void updateMotorSpeeds() {
  const MotionProfile& profile = currentMotionProfile();
  float accel_global_scale = 0.75f + 0.25f * speed_multiplier;

  for (int i = 0; i < 6; i++) {
    JOINTS[i]->setMaxSpeed(BASE_MAX_SPEEDS[i] * speed_multiplier * profile.speed_scale * JOINT_SPEED_SCALES[i]);
    JOINTS[i]->setAcceleration(BASE_MAX_ACCELS[i] * accel_global_scale * profile.accel_scale * JOINT_ACCEL_SCALES[i]);
  }
}

const MotionProfile& currentMotionProfile() {
  if (motion_profile_mode > PROFILE_FAST) {
    motion_profile_mode = PROFILE_BALANCED;
  }
  return MOTION_PROFILES[motion_profile_mode];
}

void clearCoordinatedMotionPlan() {
  coordinated_plan_active = false;
  active_lead_distance = 0;
  updateMotorSpeeds();
}

void refreshActiveMoveSpeeds() {
  if (is_moving && coordinated_plan_active) {
    beginCoordinatedMove(active_move_targets);
  } else {
    updateMotorSpeeds();
  }
}

void beginCoordinatedMove(const long* targetSteps) {
  const MotionProfile& profile = currentMotionProfile();
  float accel_global_scale = 0.75f + 0.25f * speed_multiplier;
  long lead_distance = 0;
  bool j3_reverse = false;

  for (int i = 0; i < 6; i++) {
    active_move_targets[i] = targetSteps[i];
    active_move_delta[i] = targetSteps[i] - JOINTS[i]->currentPosition();
    long abs_delta = active_move_delta[i];
    if (abs_delta < 0) abs_delta = -abs_delta;
    if (abs_delta > lead_distance) {
      lead_distance = abs_delta;
    }
  }

  if (previous_move_delta[2] != 0 && active_move_delta[2] != 0) {
    j3_reverse = ((previous_move_delta[2] > 0) != (active_move_delta[2] > 0));
  }

  if (lead_distance == 0) {
    clearCoordinatedMotionPlan();
    return;
  }

  float short_move_ratio = 1.0f;
  if (lead_distance < SHORT_MOVE_THRESHOLD_STEPS) {
    short_move_ratio = (float)lead_distance / (float)SHORT_MOVE_THRESHOLD_STEPS;
  }

  float short_speed_scale =
    SHORT_MOVE_SPEED_SCALE_MIN + (1.0f - SHORT_MOVE_SPEED_SCALE_MIN) * short_move_ratio;
  float short_accel_scale =
    SHORT_MOVE_ACCEL_SCALE_MIN + (1.0f - SHORT_MOVE_ACCEL_SCALE_MIN) * short_move_ratio;
  float short_floor_scale =
    SHORT_MOVE_FLOOR_SCALE_MIN + (1.0f - SHORT_MOVE_FLOOR_SCALE_MIN) * short_move_ratio;

  float lead_speed_cap = 1.0e9f;
  float lead_accel_cap = 1.0e9f;

  for (int i = 0; i < 6; i++) {
    long abs_delta = active_move_delta[i];
    if (abs_delta < 0) abs_delta = -abs_delta;
    if (abs_delta == 0) {
      continue;
    }

    float ratio = (float)abs_delta / (float)lead_distance;
    float axis_speed_cap = BASE_MAX_SPEEDS[i] * speed_multiplier * profile.speed_scale * JOINT_SPEED_SCALES[i] * short_speed_scale;
    float axis_accel_cap = BASE_MAX_ACCELS[i] * accel_global_scale * profile.accel_scale * JOINT_ACCEL_SCALES[i] * short_accel_scale;
    if (i == 2 && j3_reverse) {
      axis_speed_cap *= J3_REVERSE_SPEED_SCALE;
      axis_accel_cap *= J3_REVERSE_ACCEL_SCALE;
    }
    if (axis_speed_cap / ratio < lead_speed_cap) {
      lead_speed_cap = axis_speed_cap / ratio;
    }
    if (axis_accel_cap / ratio < lead_accel_cap) {
      lead_accel_cap = axis_accel_cap / ratio;
    }
  }

  coordinated_plan_active = true;
  active_lead_distance = lead_distance;

  for (int i = 0; i < 6; i++) {
    JOINTS[i]->moveTo(active_move_targets[i]);

    long abs_delta = active_move_delta[i];
    if (abs_delta < 0) abs_delta = -abs_delta;
    if (abs_delta == 0) {
      previous_move_delta[i] = 0;
      active_move_speeds[i] = 0.0f;
      active_move_accels[i] = 0.0f;
      continue;
    }

    float ratio = (float)abs_delta / (float)lead_distance;
    float axis_speed_cap = BASE_MAX_SPEEDS[i] * speed_multiplier * profile.speed_scale * JOINT_SPEED_SCALES[i] * short_speed_scale;
    float axis_accel_cap = BASE_MAX_ACCELS[i] * accel_global_scale * profile.accel_scale * JOINT_ACCEL_SCALES[i] * short_accel_scale;
    float axis_min_speed = COORD_MIN_SPEEDS[i] * speed_multiplier * JOINT_MIN_SPEED_SCALES[i] * short_floor_scale;
    float axis_min_accel = COORD_MIN_ACCELS[i] * accel_global_scale * JOINT_MIN_ACCEL_SCALES[i] * short_floor_scale;
    if (i == 2 && j3_reverse) {
      axis_speed_cap *= J3_REVERSE_SPEED_SCALE;
      axis_accel_cap *= J3_REVERSE_ACCEL_SCALE;
    }

    active_move_speeds[i] = lead_speed_cap * ratio;
    if (active_move_speeds[i] < axis_min_speed) {
      active_move_speeds[i] = axis_min_speed;
    }
    if (active_move_speeds[i] > axis_speed_cap) {
      active_move_speeds[i] = axis_speed_cap;
    }

    active_move_accels[i] = lead_accel_cap * ratio;
    if (active_move_accels[i] < axis_min_accel) {
      active_move_accels[i] = axis_min_accel;
    }
    if (active_move_accels[i] > axis_accel_cap) {
      active_move_accels[i] = axis_accel_cap;
    }

    JOINTS[i]->setMaxSpeed(active_move_speeds[i]);
    JOINTS[i]->setAcceleration(active_move_accels[i]);
    previous_move_delta[i] = active_move_delta[i];
  }

  Serial.print(F("[PLAN] profile="));
  Serial.print(profile.name);
  Serial.print(F(" lead_steps="));
  Serial.print(active_lead_distance);
  Serial.print(F(" lead_v="));
  Serial.print(lead_speed_cap, 1);
  Serial.print(F(" lead_a="));
  Serial.print(lead_accel_cap, 1);
  Serial.print(F(" j3_reverse="));
  Serial.print(j3_reverse ? F("YES") : F("NO"));
  Serial.print(F(" short_v="));
  Serial.print(short_speed_scale, 2);
  Serial.print(F(" short_a="));
  Serial.println(short_accel_scale, 2);
}

int16_t encode_task_pos_mm(float meters) {
  float scaled = meters * 1000.0f;
  if (scaled > 32767.0f) scaled = 32767.0f;
  if (scaled < -32768.0f) scaled = -32768.0f;
  return (int16_t)(scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f);
}

float decode_task_pos_mm(int16_t millimeters) {
  return (float)millimeters / 1000.0f;
}

int16_t encode_task_angle_deg10(float degrees) {
  float scaled = degrees * 10.0f;
  if (scaled > 32767.0f) scaled = 32767.0f;
  if (scaled < -32768.0f) scaled = -32768.0f;
  return (int16_t)(scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f);
}

float decode_task_angle_deg10(int16_t deg10) {
  return (float)deg10 / 10.0f;
}

float decode_task_angle_rad(int16_t deg10) {
  return decode_task_angle_deg10(deg10) * PI / 180.0f;
}

uint16_t encode_task_time_cs(float seconds) {
  if (seconds < 0.0f) seconds = 0.0f;
  float scaled = seconds * 100.0f;
  if (scaled > 65535.0f) scaled = 65535.0f;
  return (uint16_t)(scaled + 0.5f);
}

float decode_task_time_cs(uint16_t centiseconds) {
  return (float)centiseconds / 100.0f;
}

bool enqueueTaskCommand(int task_idx,
                        float x, float y, float z,
                        float r, float p, float yaw,
                        float delay_s, float j6_extra_deg,
                        int8_t d3_state, float gripper_delay_s) {
  if (task_idx < 0 || task_idx >= TASK_LIST_COUNT) {
    return false;
  }
  if (task_count[task_idx] >= MAX_TASK_ITEMS) {
    Serial.print(F("ERROR: Task list "));
    Serial.print(task_idx + 1);
    Serial.println(F(" full"));
    return false;
  }

  int idx = task_count[task_idx];
  if (delay_s < 0.0f) delay_s = 0.0f;
  if (gripper_delay_s < 0.0f) gripper_delay_s = 0.0f;
  task_p_mm[task_idx][idx][0] = encode_task_pos_mm(x);
  task_p_mm[task_idx][idx][1] = encode_task_pos_mm(y);
  task_p_mm[task_idx][idx][2] = encode_task_pos_mm(z);
  task_rpy_deg10[task_idx][idx][0] = encode_task_angle_deg10(r);
  task_rpy_deg10[task_idx][idx][1] = encode_task_angle_deg10(p);
  task_rpy_deg10[task_idx][idx][2] = encode_task_angle_deg10(yaw);
  task_delay_cs[task_idx][idx] = encode_task_time_cs(delay_s);
  task_j6_extra_deg10[task_idx][idx] = encode_task_angle_deg10(j6_extra_deg);
  task_d3_state[task_idx][idx] = d3_state;
  task_gripper_delay_cs[task_idx][idx] = encode_task_time_cs(gripper_delay_s);
  task_count[task_idx]++;

  Serial.print(F("Stored G"));
  Serial.print(task_idx + 1);
  Serial.print(F(" #"));
  Serial.print(idx);
  Serial.print(F(" P=("));
  Serial.print(x, 3); Serial.print(F(","));
  Serial.print(y, 3); Serial.print(F(","));
  Serial.print(z, 3); Serial.print(F(") RPY=("));
  Serial.print(r, 1); Serial.print(F(","));
  Serial.print(p, 1); Serial.print(F(","));
  Serial.print(yaw, 1); Serial.print(F(")"));
  if (delay_s > 0.0f) {
    Serial.print(F(" delay="));
    Serial.print(delay_s, 2);
  }
  if (j6_extra_deg != 0.0f) {
    Serial.print(F(" J6+="));
    Serial.print(j6_extra_deg, 1);
    Serial.print(F("deg"));
  }
  if (d3_state == 1) Serial.print(F(" D3=HIGH"));
  else if (d3_state == 0) Serial.print(F(" D3=LOW"));
  if (gripper_delay_s > 0.0f) {
    Serial.print(F(" grip_delay="));
    Serial.print(gripper_delay_s, 1);
    Serial.print(F("s"));
  }
  Serial.println();
  return true;
}

bool startTaskList(int task_idx, const char* source_name, bool quiet_failures) {
  if (task_idx < 0 || task_idx >= TASK_LIST_COUNT) {
    return false;
  }
  if (task_count[task_idx] == 0) {
    if (!quiet_failures) {
      Serial.print(F("ERROR: Task list "));
      Serial.print(task_idx + 1);
      Serial.println(F(" empty"));
    }
    return false;
  }
  if (is_moving || waiting_for_delay || waiting_for_gripper || control_mode != MODE_IDLE) {
    if (!quiet_failures) {
      Serial.print(F("BUSY: Cannot start task list "));
      Serial.print(task_idx + 1);
      Serial.print(F(" from "));
      Serial.println(source_name);
    }
    return false;
  }

  queue_count = task_count[task_idx];
  for (int i = 0; i < queue_count; i++) {
    queue_p[i][0] = decode_task_pos_mm(task_p_mm[task_idx][i][0]);
    queue_p[i][1] = decode_task_pos_mm(task_p_mm[task_idx][i][1]);
    queue_p[i][2] = decode_task_pos_mm(task_p_mm[task_idx][i][2]);
    queue_rpy[i][0] = decode_task_angle_rad(task_rpy_deg10[task_idx][i][0]);
    queue_rpy[i][1] = decode_task_angle_rad(task_rpy_deg10[task_idx][i][1]);
    queue_rpy[i][2] = decode_task_angle_rad(task_rpy_deg10[task_idx][i][2]);
    queue_delay_s[i] = decode_task_time_cs(task_delay_cs[task_idx][i]);
    queue_j6_extra[i] = decode_task_angle_rad(task_j6_extra_deg10[task_idx][i]);
    queue_auto_orient[i] = false;
    queue_d3_state[i] = task_d3_state[task_idx][i];
    queue_gripper_delay_s[i] = decode_task_time_cs(task_gripper_delay_cs[task_idx][i]);
  }

  queue_index = 0;
  queue_running = true;
  queue_step_active = false;
  control_mode = MODE_QUEUE;

  Serial.print(F("TASK "));
  Serial.print(task_idx + 1);
  Serial.print(F(" RUN from "));
  Serial.print(source_name);
  Serial.print(F(" with "));
  Serial.print(queue_count);
  Serial.println(F(" commands"));
  return true;
}

void clearTaskList(int task_idx) {
  if (task_idx < 0 || task_idx >= TASK_LIST_COUNT) {
    return;
  }
  task_count[task_idx] = 0;
  Serial.print(F("Task list "));
  Serial.print(task_idx + 1);
  Serial.println(F(" cleared"));
}

void printTaskList(int task_idx) {
  if (task_idx < 0 || task_idx >= TASK_LIST_COUNT) {
    return;
  }
  Serial.print(F("Task list "));
  Serial.print(task_idx + 1);
  Serial.print(F(": "));
  Serial.print(task_count[task_idx]);
  Serial.println(F(" commands"));
  for (int i = 0; i < task_count[task_idx]; i++) {
    Serial.print(F("  #"));
    Serial.print(i);
    Serial.print(F(" P=("));
    Serial.print(decode_task_pos_mm(task_p_mm[task_idx][i][0]), 3); Serial.print(F(","));
    Serial.print(decode_task_pos_mm(task_p_mm[task_idx][i][1]), 3); Serial.print(F(","));
    Serial.print(decode_task_pos_mm(task_p_mm[task_idx][i][2]), 3); Serial.print(F(") RPY=("));
    Serial.print(decode_task_angle_deg10(task_rpy_deg10[task_idx][i][0]), 1); Serial.print(F(","));
    Serial.print(decode_task_angle_deg10(task_rpy_deg10[task_idx][i][1]), 1); Serial.print(F(","));
    Serial.print(decode_task_angle_deg10(task_rpy_deg10[task_idx][i][2]), 1); Serial.print(F(")"));
    Serial.print(F(" delay="));
    Serial.print(decode_task_time_cs(task_delay_cs[task_idx][i]), 2);
    Serial.println(F("s"));
  }
}

void checkTaskTriggerInputs() {
  unsigned long now = millis();
  for (int i = 0; i < TASK_LIST_COUNT; i++) {
    bool raw_state = (digitalRead(TASK_TRIGGER_PINS[i]) == HIGH);
    if (raw_state != task_trigger_raw_state[i]) {
      task_trigger_raw_state[i] = raw_state;
      task_trigger_last_change_ms[i] = now;
    }

    if ((now - task_trigger_last_change_ms[i]) < TASK_TRIGGER_DEBOUNCE_MS) {
      continue;
    }

    if (task_trigger_prev_state[i] != task_trigger_raw_state[i]) {
      task_trigger_prev_state[i] = task_trigger_raw_state[i];
      if (!task_trigger_prev_state[i]) {
        task_trigger_latched[i] = false;
      }
    }

    if (!task_trigger_prev_state[i] || task_trigger_latched[i]) {
      continue;
    }

    task_trigger_latched[i] = true;
    if (startTaskList(i, TASK_TRIGGER_NAMES[i], true)) {
        return;
    }
  }
}

// ============================================================
// SERIAL COMMAND PARSER
// ============================================================
void parseSerialCommand() {
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    
    if (cmd.length() == 0) return;
    
    String cmdUpper = cmd;
    cmdUpper.toUpperCase();
    
    // First command char
    char firstChar = cmdUpper.charAt(0);
    if (firstChar == 'G' || firstChar == 'g') {
      bool is_circle = cmdUpper.startsWith("GC");
      bool use_radian = false;
      int task_idx = -1;
      if (cmdUpper.startsWith("G1 ")) task_idx = 0;
      else if (cmdUpper.startsWith("G2 ")) task_idx = 1;
      else if (cmdUpper.startsWith("G3 ")) task_idx = 2;
      if (is_circle) {
        if (cmdUpper.startsWith("GCR")) use_radian = true;
      } else {
        if (cmdUpper.startsWith("GR")) use_radian = true;
      }

      if (task_idx >= 0) {
        float values[10];
        int valueCount = 0;
        int startPos = cmd.indexOf(' ');
        if (startPos == -1) {
          Serial.println(F("ERROR: No parameters found"));
          return;
        }

        String remaining = cmd.substring(startPos);
        remaining.trim();

        while (remaining.length() > 0 && valueCount < 10) {
          int spacePos = remaining.indexOf(' ');
          String token;
          if (spacePos == -1) {
            token = remaining;
            remaining = "";
          } else {
            token = remaining.substring(0, spacePos);
            remaining = remaining.substring(spacePos + 1);
            remaining.trim();
          }

          if (token.length() > 0) {
            values[valueCount] = token.toFloat();
            valueCount++;
          }
        }

        if (valueCount < 6 || valueCount > 10) {
          Serial.print(F("ERROR: Invalid format for G"));
          Serial.print(task_idx + 1);
          Serial.print(F(". Found "));
          Serial.print(valueCount);
          Serial.println(F(" parameters"));
          Serial.println(F("Format: G1/G2/G3 x y z roll pitch yaw [delay_s] [j6_deg] [d3] [gripper_delay_s]"));
          return;
        }

        float delay_s = (valueCount >= 7) ? values[6] : 0.0f;
        float j6_extra_deg = (valueCount >= 8) ? values[7] : 0.0f;
        int8_t d3_state = (valueCount >= 9) ? (int8_t)values[8] : -1;
        float gripper_delay = (valueCount == 10) ? values[9] : 0.0f;
        enqueueTaskCommand(task_idx,
                           values[0], values[1], values[2],
                           values[3], values[4], values[5],
                           delay_s, j6_extra_deg,
                           d3_state, gripper_delay);
        return;
      }

      // Parse numbers
      float values[11];
      int valueCount = 0;
      int maxCount = is_circle ? 7 : 10; // 10 for G/GR: x y z r p yaw [delay_s] [j6_deg] [d3] [gripper_delay_s]

      int startPos = cmd.indexOf(' ');
      if (startPos == -1) {
        Serial.println(F("ERROR: No parameters found"));
        return;
      }

      String remaining = cmd.substring(startPos);
      remaining.trim();

      while (remaining.length() > 0 && valueCount < maxCount) {
        int spacePos = remaining.indexOf(' ');
        String token;
        if (spacePos == -1) {
          token = remaining;
          remaining = "";
        } else {
          token = remaining.substring(0, spacePos);
          remaining = remaining.substring(spacePos + 1);
          remaining.trim();
        }

        if (token.length() > 0) {
          values[valueCount] = token.toFloat();
          valueCount++;
        }
      }

      if (!is_circle && (valueCount >= 6 && valueCount <= 10)) {
        float x = values[0];
        float y = values[1];
        float z = values[2];
        float r = values[3];
        float p = values[4];
        float yaw = values[5];
        float delay_s         = (valueCount >= 7)  ? values[6] : 0.0f;
        float j6_extra_deg    = (valueCount >= 8)  ? values[7] : 0.0f;
        int8_t d3_state       = (valueCount >= 9)  ? (int8_t)values[8] : -1;
        float gripper_delay   = (valueCount == 10) ? values[9] : 0.0f; // chờ trước khi kích gripper

        if (queue_count >= MAX_QUEUE) {
          Serial.println(F("ERROR: Queue full"));
          return;
        }

        queue_p[queue_count][0] = x;
        queue_p[queue_count][1] = y;
        queue_p[queue_count][2] = z;

        last_g_center[0] = x;
        last_g_center[1] = y;
        last_g_center[2] = z;
        last_g_center_valid = true;

        if (delay_s < 0.0f) delay_s = 0.0f;
        queue_delay_s[queue_count] = delay_s;
        queue_auto_orient[queue_count] = false; // Normal G/GR mode
        queue_d3_state[queue_count] = d3_state;
        queue_gripper_delay_s[queue_count] = gripper_delay;

        if (use_radian) {
          queue_rpy[queue_count][0] = r;
          queue_rpy[queue_count][1] = p;
          queue_rpy[queue_count][2] = yaw;
          queue_j6_extra[queue_count] = j6_extra_deg; // Already in radians if GR
          Serial.print(F("Queued G (rad) #"));
        } else {
          queue_rpy[queue_count][0] = r * PI / 180.0;
          queue_rpy[queue_count][1] = p * PI / 180.0;
          queue_rpy[queue_count][2] = yaw * PI / 180.0;
          queue_j6_extra[queue_count] = j6_extra_deg * PI / 180.0; // Convert to radians
          Serial.print(F("Queued G (deg) #"));
        }
        Serial.print(queue_count);
        Serial.print(F(" delay="));
        Serial.print(queue_delay_s[queue_count], 2);
        if (j6_extra_deg != 0) {
          Serial.print(F("s J6+="));
          Serial.print(j6_extra_deg, 1);
          Serial.print(F("deg"));
        }
        if (d3_state == 1)       Serial.print(F(" D3=HIGH"));
        else if (d3_state == 0)  Serial.print(F(" D3=LOW"));
        if (gripper_delay > 0) {
          Serial.print(F(" grip_delay="));
          Serial.print(gripper_delay, 1);
          Serial.print(F("s"));
        }
        Serial.println();
        queue_count++;
      } else if (is_circle && (valueCount == 6 || valueCount == 7)) {
        float z = values[0];
        float r = values[1];
        float p = values[2];
        float yaw = values[3];
        float radius_in = values[4];
        float angle = values[5];
        float delay_s = (valueCount == 7) ? values[6] : 0.0f;

        if (queue_count + ARC_STEPS > MAX_QUEUE) {
          Serial.println(F("ERROR: Queue full for GC"));
          return;
        }

        if (!last_g_center_valid) {
          Serial.println(F("ERROR: GC needs center from last G/GR"));
          return;
        }

        if (!use_radian) {
          r *= PI / 180.0f;
          p *= PI / 180.0f;
          yaw *= PI / 180.0f;
          angle *= PI / 180.0f;
        }

        float cx = last_g_center[0];
        float cy = last_g_center[1];

        if (queue_count == 0) {
          Serial.println(F("ERROR: GC needs a start point in queue"));
          return;
        }
        float start_x = queue_p[queue_count - 1][0];
        float start_y = queue_p[queue_count - 1][1];

        float dx0 = start_x - cx;
        float dy0 = start_y - cy;
        float radius = sqrtf(dx0*dx0 + dy0*dy0);
        if (radius_in > 0.0f) radius = radius_in;
        if (radius < 1e-6f) {
          Serial.println(F("ERROR: GC radius too small"));
          return;
        }

        float start_ang = (fabsf(dx0) < 1e-6f && fabsf(dy0) < 1e-6f) ? 0.0f : atan2f(dy0, dx0);
        float end_x = cx + radius * cosf(start_ang + angle);
        float end_y = cy + radius * sinf(start_ang + angle);

        if (delay_s < 0.0f) delay_s = 0.0f;

        for (int i = 1; i <= ARC_STEPS; i++) {
          float t = (float)i / (float)ARC_STEPS;
          float theta = start_ang + angle * t;
          float px = cx + radius * cosf(theta);
          float py = cy + radius * sinf(theta);

          queue_p[queue_count][0] = px;
          queue_p[queue_count][1] = py;
          queue_p[queue_count][2] = z;
          queue_rpy[queue_count][0] = r;
          queue_rpy[queue_count][1] = p;
          queue_rpy[queue_count][2] = yaw;
          queue_delay_s[queue_count] = (i == ARC_STEPS) ? delay_s : 0.0f;
          queue_j6_extra[queue_count] = 0; // GC doesn't support J6 extra yet
          queue_auto_orient[queue_count] = false; // GC mode uses fixed orientation
          queue_d3_state[queue_count] = -1; // GC không thay đổi D3
          queue_gripper_delay_s[queue_count] = 0;
          queue_count++;
        }

        Serial.print(F("Queued GC arc with "));
        Serial.print(ARC_STEPS);
        Serial.println(F(" points"));
      } else {
        Serial.print(F("ERROR: Invalid format for command: '"));
        Serial.print(cmd);
        Serial.print(F("' (Found "));
        Serial.print(valueCount);
        Serial.println(F(" parameters)"));
        Serial.println(F("Format should be:"));
        Serial.println(F("  G x y z roll pitch yaw [delay_s]"));
        Serial.println(F("  GR x y z roll pitch yaw [delay_s]"));
        Serial.println(F("  GC z r p y radius angle [delay_s]"));
        Serial.println(F("  GCR z r p y radius angle [delay_s]"));
      }
    }
    else if (cmd.startsWith("IK0") || cmd.startsWith("ik0") || cmd.startsWith("Ik0") || cmd.startsWith("iK0")) {
      // IK0 mode: Auto-find orientation
      // Format: IK0 x y z [delay_s] [j6_extra_deg]
      float values[5];
      int valueCount = 0;

      int startPos = cmd.indexOf(' ');
      if (startPos == -1) {
        Serial.println(F("ERROR: No parameters found"));
        return;
      }

      String remaining = cmd.substring(startPos);
      remaining.trim();

      while (remaining.length() > 0 && valueCount < 5) {
        int spacePos = remaining.indexOf(' ');
        String token;
        if (spacePos == -1) {
          token = remaining;
          remaining = "";
        } else {
          token = remaining.substring(0, spacePos);
          remaining = remaining.substring(spacePos + 1);
          remaining.trim();
        }

        if (token.length() > 0) {
          values[valueCount] = token.toFloat();
          valueCount++;
        }
      }

      if (valueCount == 3 || valueCount == 4 || valueCount == 5) {
        float x = values[0];
        float y = values[1];
        float z = values[2];
        float delay_s = (valueCount >= 4) ? values[3] : 0.0f;
        float j6_extra_deg = (valueCount == 5) ? values[4] : 0.0f; // Extra J6 in degrees

        if (queue_count >= MAX_QUEUE) {
          Serial.println(F("ERROR: Queue full"));
          return;
        }

        queue_p[queue_count][0] = x;
        queue_p[queue_count][1] = y;
        queue_p[queue_count][2] = z;
        queue_j6_extra[queue_count] = j6_extra_deg * PI / 180.0; // Convert to radians
        
        // Tìm hướng hợp lệ NGAY BÂY GIỜ
        Serial.println(F("IK0: Searching..."));
        
        float q_temp[6];
        float found_rpy[3];
        bool found = ik6dof_auto_orient(x, y, z, q_temp, found_rpy);
        
        if (found) {
          // Tìm thấy hướng hợp lệ! Lưu vào queue
          queue_rpy[queue_count][0] = found_rpy[0];
          queue_rpy[queue_count][1] = found_rpy[1];
          queue_rpy[queue_count][2] = found_rpy[2];
          queue_auto_orient[queue_count] = false; // Không cần auto nữa vì đã có hướng
          queue_d3_state[queue_count] = -1;
          queue_gripper_delay_s[queue_count] = 0;

          Serial.print(F("Found RPY=("));
          Serial.print(found_rpy[0]*180/PI, 1); Serial.print(F(","));
          Serial.print(found_rpy[1]*180/PI, 1); Serial.print(F(","));
          Serial.print(found_rpy[2]*180/PI, 1); Serial.println(F(")"));
        } else {
          // Không tìm thấy hướng hợp lệ
          Serial.println(F("ERROR: No valid orientation!"));
          return; // Không thêm vào queue
        }

        last_g_center[0] = x;
        last_g_center[1] = y;
        last_g_center[2] = z;
        last_g_center_valid = true;

        if (delay_s < 0.0f) delay_s = 0.0f;
        queue_delay_s[queue_count] = delay_s;

        Serial.print(F("Queued #"));
        Serial.print(queue_count);
        Serial.print(F(": P=("));
        Serial.print(x, 3); Serial.print(F(","));
        Serial.print(y, 3); Serial.print(F(","));
        Serial.print(z, 3); Serial.print(F(") RPY=("));
        Serial.print(found_rpy[0]*180/PI, 1); Serial.print(F(" deg,"));
        Serial.print(found_rpy[1]*180/PI, 1); Serial.print(F(" deg,"));
        Serial.print(found_rpy[2]*180/PI, 1); Serial.print(F(" deg) T="));
        Serial.print(delay_s, 1);
        Serial.println(F("s"));
        queue_count++;
      } else {
        Serial.print(F("ERROR: Invalid format for IK0. Found "));
        Serial.print(valueCount);
        Serial.println(F(" parameters"));
        Serial.println(F("Format: IK0 x y z [delay_s]"));
        Serial.println(F("Example: IK0 0.4 0.1 0.5 1.5"));
      }
    }
    else if (firstChar == 'H' || firstChar == 'h') {
      // Home position
      target_p[0] = 0.0;
      target_p[1] = 0.0;
      target_p[2] = 0.88057;
      target_rpy[0] = 0;
      target_rpy[1] = 0;
      target_rpy[2] = 0;
      has_new_target = true;
      control_mode = MODE_MANUAL;
      Serial.println(F("Going HOME..."));
    }
    else if (cmdUpper == "A1" || cmdUpper == "A2" || cmdUpper == "A3") {
      startTaskList(cmdUpper.charAt(1) - '1', "serial", false);
    }
    else if (firstChar == 'A' || firstChar == 'a') {
      // Run queued G/GR commands
      if (queue_count == 0) {
        Serial.println(F("ERROR: Queue empty"));
        return;
      }
      for (int i = 0; i < 6; i++) { q_prev[i] = HOME_Q_IK[i]; q_current_ik[i] = HOME_Q_IK[i]; }
      queue_index = 0;
      queue_running = true;
      queue_step_active = false;
      control_mode = MODE_QUEUE;
      Serial.print(F("QUEUE RUN: "));
      Serial.print(queue_count);
      Serial.println(F(" targets"));
    }
    else if (firstChar == 'S' || firstChar == 's') {
      // Stop/Idle
      control_mode = MODE_IDLE;
      queue_running = false;
      queue_step_active = false;
      Serial.println(F("IDLE mode: Stopped"));
    }
    else if (cmdUpper == "L1" || cmdUpper == "L2" || cmdUpper == "L3") {
      printTaskList(cmdUpper.charAt(1) - '1');
    }
    else if (firstChar == 'L' || firstChar == 'l') {
      // List queue
      Serial.print(F("Queue count: "));
      Serial.println(queue_count);
      for (int i = 0; i < queue_count; i++) {
        Serial.print(F("Q"));
        Serial.print(i);
        Serial.print(F(": P=("));
        Serial.print(queue_p[i][0], 3); Serial.print(F(","));
        Serial.print(queue_p[i][1], 3); Serial.print(F(","));
        Serial.print(queue_p[i][2], 3); Serial.print(F(") RPY=("));
        Serial.print(queue_rpy[i][0], 4); Serial.print(F(","));
        Serial.print(queue_rpy[i][1], 4); Serial.print(F(","));
        Serial.print(queue_rpy[i][2], 4);
        Serial.print(F(") delay="));
        Serial.print(queue_delay_s[i], 2);
        Serial.println(F("s"));
      }
    }
    else if (cmdUpper == "C1" || cmdUpper == "C2" || cmdUpper == "C3") {
      clearTaskList(cmdUpper.charAt(1) - '1');
    }
    else if (firstChar == 'C' || firstChar == 'c') {
      // Clear queue
      queue_count = 0;
      queue_index = 0;
      queue_running = false;
      queue_step_active = false;
      for (int i = 0; i < 6; i++) { q_prev[i] = HOME_Q_IK[i]; q_current_ik[i] = HOME_Q_IK[i]; }
      for (int i = 0; i < MAX_QUEUE; i++) {
        queue_auto_orient[i] = false;
        queue_j6_extra[i] = 0;
        queue_d3_state[i] = -1;
        queue_gripper_delay_s[i] = 0;
      }
      Serial.println(F("Queue cleared"));
    }
    else if (firstChar == '?') {
      // Help
      printHelp();
    }
    else if (firstChar == 'V' || firstChar == 'v') {
      // Speed control: V <percentage>
      // Ví dụ: V 150 (tăng 150%), V 50 (giảm 50%)
      int spacePos = cmd.indexOf(' ');
      if (spacePos != -1) {
        String speedStr = cmd.substring(spacePos + 1);
        speedStr.trim();
        float newSpeed = speedStr.toFloat();
        
        if (newSpeed >= SPEED_MIN * 100 && newSpeed <= SPEED_MAX * 100) {
          const MotionProfile& profile = currentMotionProfile();
          speed_multiplier = newSpeed / 100.0;
          refreshActiveMoveSpeeds();
          Serial.print(F("Speed set to "));
          Serial.print(newSpeed, 0);
          Serial.print(F("% (multiplier="));
          Serial.print(speed_multiplier, 2);
          Serial.println(F(")"));
          Serial.print(F("Motion profile: "));
          Serial.println(profile.name);
          Serial.print(F("Motor speeds: J1="));
          Serial.print((int)(BASE_MAX_SPEEDS[0] * speed_multiplier * profile.speed_scale * JOINT_SPEED_SCALES[0]));
          Serial.print(F(" J2="));
          Serial.print((int)(BASE_MAX_SPEEDS[1] * speed_multiplier * profile.speed_scale * JOINT_SPEED_SCALES[1]));
          Serial.print(F(" J3="));
          Serial.print((int)(BASE_MAX_SPEEDS[2] * speed_multiplier * profile.speed_scale * JOINT_SPEED_SCALES[2]));
          Serial.print(F(" J4="));
          Serial.print((int)(BASE_MAX_SPEEDS[3] * speed_multiplier * profile.speed_scale * JOINT_SPEED_SCALES[3]));
          Serial.print(F(" J5="));
          Serial.print((int)(BASE_MAX_SPEEDS[4] * speed_multiplier * profile.speed_scale * JOINT_SPEED_SCALES[4]));
          Serial.print(F(" J6="));
          Serial.print((int)(BASE_MAX_SPEEDS[5] * speed_multiplier * profile.speed_scale * JOINT_SPEED_SCALES[5]));
          Serial.println(F(" steps/s"));
          float accel_global_scale = 0.75f + 0.25f * speed_multiplier;
          Serial.print(F("Motor accels: J1="));
          Serial.print((int)(BASE_MAX_ACCELS[0] * accel_global_scale * profile.accel_scale * JOINT_ACCEL_SCALES[0]));
          Serial.print(F(" J2="));
          Serial.print((int)(BASE_MAX_ACCELS[1] * accel_global_scale * profile.accel_scale * JOINT_ACCEL_SCALES[1]));
          Serial.print(F(" J3="));
          Serial.print((int)(BASE_MAX_ACCELS[2] * accel_global_scale * profile.accel_scale * JOINT_ACCEL_SCALES[2]));
          Serial.print(F(" J4="));
          Serial.print((int)(BASE_MAX_ACCELS[3] * accel_global_scale * profile.accel_scale * JOINT_ACCEL_SCALES[3]));
          Serial.print(F(" J5="));
          Serial.print((int)(BASE_MAX_ACCELS[4] * accel_global_scale * profile.accel_scale * JOINT_ACCEL_SCALES[4]));
          Serial.print(F(" J6="));
          Serial.print((int)(BASE_MAX_ACCELS[5] * accel_global_scale * profile.accel_scale * JOINT_ACCEL_SCALES[5]));
          Serial.println(F(" steps/s^2"));
        } else {
          Serial.print(F("ERROR: Speed must be between "));
          Serial.print(SPEED_MIN * 100, 0);
          Serial.print(F("% and "));
          Serial.print(SPEED_MAX * 100, 0);
          Serial.println(F("%"));
        }
      } else {
        // Hiển thị tốc độ hiện tại
        Serial.print(F("Current speed: "));
        Serial.print(speed_multiplier * 100, 0);
        Serial.println(F("%"));
        Serial.print(F("Motion profile: "));
        Serial.println(currentMotionProfile().name);
        Serial.println(F("Usage: V <percentage>"));
        Serial.println(F("Example: V 150 (increase to 150%)"));
        Serial.println(F("Example: V 50 (decrease to 50%)"));
      }
    }
    else {
      Serial.print(F("ERROR: Unknown command: '"));
      Serial.print(cmd);
      Serial.println(F("'"));
      Serial.println(F("Type ? for help"));
    }
  }
}

// ============================================================
// HELP MENU
// ============================================================
void printHelp() {
  Serial.println(F("\n===== ROBOT 6DOF COMMANDS ====="));
  Serial.println(F("G x y z r p y [delay] [j6] [d3] [gdelay] - Queue (DEGREES)"));
  Serial.println(F("G1/G2/G3 ... - Store command to task list 1/2/3"));
  Serial.println(F("  d3: 1=HIGH(dong), 0=LOW(mo), bo trong=giu"));
  Serial.println(F("  gdelay: giay cho truoc khi kich gripper"));
  Serial.println(F("  Ex: G 0.35 0 0.16 0 180 0 1 90 1 2"));
  Serial.println(F("  Ex: G1 0.32 0 0.33 0 120 0 0 0 0"));
  Serial.println(F("GR x y z r p y [delay] [j6] [d3] [gdelay] - Queue (RADIANS)"));
  Serial.println(F("  Ex: GR 0.4 0.1 0.5 0 1.5708 0 1.5 0 1"));
  Serial.println(F("IK0 x y z [delay] [j6]  - Queue AUTO orientation"));
  Serial.println(F("GC  z r p y R ang [delay] - Arc (degrees)"));
  Serial.println(F("GCR z r p y R ang [delay] - Arc (radians)"));
  Serial.println(F("H   - HOME (0,0,0.855)"));
  Serial.println(F("A   - Run queue"));
  Serial.println(F("A1/A2/A3 - Run task list 1/2/3"));
  Serial.println(F("S   - Stop"));
  Serial.println(F("L   - List queue"));
  Serial.println(F("L1/L2/L3 - List task list 1/2/3"));
  Serial.println(F("C   - Clear queue"));
  Serial.println(F("C1/C2/C3 - Clear task list 1/2/3"));
  Serial.println(F("V % - Speed (10-300%)"));
  Serial.println(F("D2/D14/D15 HIGH - Trigger task list 1/2/3"));
  Serial.println(F("?   - Help"));
  Serial.println(F("==============================="));
}

// ============================================================
// SETUP
// ============================================================
void setup() {
  Serial.begin(115200);
  Serial.println(F("=== ROBOT 6DOF STANDALONE MODE ==="));
  Serial.println(F("Initializing..."));
  
  pinMode(13, OUTPUT);
  pinMode(GRIPPER_PIN, OUTPUT);
  digitalWrite(GRIPPER_PIN, LOW);  // Mặc định mở gripper
  
  // Enable all motor drivers (LOW = enabled for RAMPS)
  pinMode(X_ENABLE_PIN, OUTPUT);
  pinMode(Z_ENABLE_PIN, OUTPUT);
  pinMode(Y_ENABLE_PIN, OUTPUT);
  pinMode(E0_ENABLE_PIN, OUTPUT);
  pinMode(E1_ENABLE_PIN, OUTPUT);
  pinMode(E2_ENABLE_PIN, OUTPUT);
  unsigned long now_ms = millis();
  for (int i = 0; i < TASK_LIST_COUNT; i++) {
    pinMode(TASK_TRIGGER_PINS[i], INPUT);
    bool trigger_state = (digitalRead(TASK_TRIGGER_PINS[i]) == HIGH);
    task_trigger_prev_state[i] = trigger_state;
    task_trigger_raw_state[i] = trigger_state;
    task_trigger_latched[i] = false;
    task_trigger_last_change_ms[i] = now_ms;
  }
  
  digitalWrite(X_ENABLE_PIN, LOW);
  digitalWrite(Z_ENABLE_PIN, LOW);
  digitalWrite(Y_ENABLE_PIN, LOW);
  digitalWrite(E0_ENABLE_PIN, LOW);
  digitalWrite(E1_ENABLE_PIN, LOW);
  digitalWrite(E2_ENABLE_PIN, LOW);
  
  // Initialize queue arrays
  for (int i = 0; i < MAX_QUEUE; i++) {
    queue_auto_orient[i] = false;
    queue_j6_extra[i] = 0;
    queue_d3_state[i] = -1;
    queue_gripper_delay_s[i] = 0;
  }
  
  // Configure steppers
  updateMotorSpeeds();  // Set initial speeds with multiplier
  for (int i = 0; i < 6; i++) {
    JOINTS[i]->setMinPulseWidth(MIN_STEP_PULSE_US);
  }
  
  // Unwrap RPY waypoints (once)
  for (int i = 0; i < 3; i++) {
    Ru_waypoints[0][i] = R_waypoints[0][i];
  }
  for (int wp = 1; wp < NUM_WAYPOINTS; wp++) {
    unwrap_vec3(Ru_waypoints[wp], R_waypoints[wp], Ru_waypoints[wp-1]);
  }
  
  Serial.println(F("Waypoints unwrapped:"));
  for (int i = 0; i < NUM_WAYPOINTS; i++) {
    Serial.print(F("WP"));
    Serial.print(i);
    Serial.print(F(": P=("));
    Serial.print(P_waypoints[i][0], 4);
    Serial.print(F(", "));
    Serial.print(P_waypoints[i][1], 4);
    Serial.print(F(", "));
    Serial.print(P_waypoints[i][2], 4);
    Serial.print(F(") RPY=("));
    Serial.print(Ru_waypoints[i][0], 4);
    Serial.print(F(", "));
    Serial.print(Ru_waypoints[i][1], 4);
    Serial.print(F(", "));
    Serial.print(Ru_waypoints[i][2], 4);
    Serial.println(F(")"));
  }
  
  Serial.println(F("Ready!"));
  Serial.println();
  Serial.println(F("Robot at HOME position (0, 0, 0.88057)"));
  Serial.println(F("Current mode: IDLE (waiting for command)"));
  Serial.print(F("Motion profile: "));
  Serial.println(currentMotionProfile().name);
  Serial.println(F("Task triggers: D2 -> G1 list, D14 -> G2 list, D15 -> G3 list"));
  Serial.println(F("Task trigger pins are active HIGH and need external pull-down resistors."));
  Serial.println();
  printHelp();
  
  digitalWrite(13, LOW);
}

// ============================================================
// LOOP
// ============================================================
void loop() {
  // Kiểm tra lệnh Serial
  parseSerialCommand();
  checkTaskTriggerInputs();
  
  float p06[3], rpy[3];
  bool should_move = false;

  // Nếu đang di chuyển, tiếp tục chạy các motor (non-blocking)
  if (is_moving) {
    // Kiểm tra xem có lệnh STOP hoặc lệnh khác không
    if (control_mode == MODE_IDLE && !queue_running) {
      // Người dùng đã nhấn S hoặc lệnh khác làm dừng queue
      digitalWrite(13, LOW);
      is_moving = false;
      clearCoordinatedMotionPlan();
      Serial.println(F("Movement stopped by user"));
      return;
    }
    
    // Planner phoi hop cac khop voi run() rieng cho tung motor
    for (int i = 0; i < 6; i++) {
      JOINTS[i]->run();
    }
    
    // Kiểm tra xem tất cả motor đã đến vị trí chưa
    bool all_done = true;
    for (int i = 0; i < 6; i++) {
      if (JOINTS[i]->distanceToGo() != 0) {
        all_done = false;
        break;
      }
    }
    
    if (all_done) {
      // Tất cả motor đã đến vị trí
      digitalWrite(13, LOW);
      is_moving = false;
      clearCoordinatedMotionPlan();

      if (control_mode == MODE_QUEUE && queue_step_active) {
        Serial.print(F("Position reached! Queue index "));
        Serial.println(queue_index);

        // Kích chân D3 theo tham số gripper của điểm này
        int8_t d3 = queue_d3_state[queue_index];
        float gdelay = queue_gripper_delay_s[queue_index];

        if (d3 != -1) {
          if (gdelay > 0.0f) {
            // Chờ gripper_delay_s rồi mới kích D3 (non-blocking)
            waiting_for_gripper = true;
            gripper_delay_start = millis();
            Serial.print(F("[GRIPPER] Se kich D3 sau "));
            Serial.print(gdelay, 1);
            Serial.println(F("s..."));
          } else {
            // Kích D3 ngay lập tức
            if (d3 == 1) {
              digitalWrite(GRIPPER_PIN, HIGH);
              Serial.println(F("Gripper D3 -> HIGH (dong)"));
            } else {
              digitalWrite(GRIPPER_PIN, LOW);
              Serial.println(F("Gripper D3 -> LOW (mo)"));
            }
          }
        }

        // Kiểm tra delay (chỉ xử lý nếu không đang chờ gripper)
        if (!waiting_for_gripper) {
          if (queue_delay_s[queue_index] > 0.0f) {
            waiting_for_delay = true;
            move_delay_start = millis();
          } else {
            queue_index++;
            queue_step_active = false;
            if (queue_index >= queue_count) {
              control_mode = MODE_IDLE;
              queue_running = false;
              Serial.println(F("QUEUE COMPLETED"));
            }
          }
        }
      }
      
      if (control_mode == MODE_MANUAL) {
        Serial.println(F("Position reached!"));
        Serial.println();
        control_mode = MODE_IDLE; // Chuyển về IDLE để sẵn sàng nhận lệnh mới
      }
    }
    return; // Quay lại đầu loop để có thể nhận lệnh mới
  }
  
  // Kiểm tra chờ gripper delay (trước khi kích D3)
  if (waiting_for_gripper) {
    if (control_mode != MODE_QUEUE) {
      waiting_for_gripper = false;
    } else {
      unsigned long elapsed = millis() - gripper_delay_start;
      float gdelay = queue_gripper_delay_s[queue_index];
      static unsigned long last_debug_ms = 0;
      if (millis() - last_debug_ms >= 500) {
        last_debug_ms = millis();
        Serial.print(F("[GRIPPER] cho: "));
        Serial.print(elapsed / 1000.0f, 1);
        Serial.print(F("s / "));
        Serial.print(gdelay, 1);
        Serial.println(F("s"));
      }
      if (elapsed >= (unsigned long)(gdelay * 1000.0f)) {
        waiting_for_gripper = false;
        // Kích D3 sau khi hết thời gian chờ
        int8_t d3 = queue_d3_state[queue_index];
        if (d3 == 1) {
          digitalWrite(GRIPPER_PIN, HIGH);
          Serial.println(F("Gripper D3 -> HIGH (dong)"));
        } else if (d3 == 0) {
          digitalWrite(GRIPPER_PIN, LOW);
          Serial.println(F("Gripper D3 -> LOW (mo)"));
        }
        // Sau khi kích gripper, xử lý delay_s như bình thường
        if (queue_delay_s[queue_index] > 0.0f) {
          waiting_for_delay = true;
          move_delay_start = millis();
        } else {
          queue_index++;
          queue_step_active = false;
          if (queue_index >= queue_count) {
            control_mode = MODE_IDLE;
            queue_running = false;
            Serial.println(F("QUEUE COMPLETED"));
          }
        }
      }
      return;
    }
  }

  // Kiểm tra delay giữa các bước trong queue
  if (waiting_for_delay) {
    // Kiểm tra xem có lệnh mới làm thay đổi control_mode không
    if (control_mode != MODE_QUEUE) {
      // Người dùng đã gửi lệnh mới (H, S, v.v.), hủy delay
      waiting_for_delay = false;
      waiting_for_gripper = false;
      queue_running = false;
      queue_step_active = false;
      Serial.println(F("Delay interrupted by new command"));
      // Không return, để xử lý lệnh mới bên dưới
    } else {
      unsigned long elapsed = millis() - move_delay_start;
      if (elapsed >= (unsigned long)(queue_delay_s[queue_index] * 1000.0f)) {
        waiting_for_delay = false;
        queue_index++;
        queue_step_active = false;
        if (queue_index >= queue_count) {
          control_mode = MODE_IDLE;
          queue_running = false;
          Serial.println(F("QUEUE COMPLETED"));
        }
      }
      return; // Tiếp tục chờ delay
    }
  }
  
  // Xử lý theo mode
  if (control_mode == MODE_QUEUE) {
    if (!queue_running) {
      queue_running = true;
      queue_step_active = false;
    }

    if (queue_index >= queue_count) {
      control_mode = MODE_IDLE;
      queue_running = false;
      Serial.println(F("QUEUE COMPLETED"));
      delay(100);
      return;
    }

    if (!queue_step_active) {
      p06[0] = queue_p[queue_index][0];
      p06[1] = queue_p[queue_index][1];
      p06[2] = queue_p[queue_index][2];
      rpy[0] = queue_rpy[queue_index][0];
      rpy[1] = queue_rpy[queue_index][1];
      rpy[2] = queue_rpy[queue_index][2];
      should_move = true;
      queue_step_active = true;
    }
  }
  else if (control_mode == MODE_MANUAL && has_new_target) {
    // MANUAL MODE: Di chuyển đến target
    p06[0] = target_p[0];
    p06[1] = target_p[1];
    p06[2] = target_p[2];
    rpy[0] = target_rpy[0];
    rpy[1] = target_rpy[1];
    rpy[2] = target_rpy[2];
    should_move = true;
    has_new_target = false;
  }
  else if (control_mode == MODE_IDLE) {
    // IDLE MODE: Không làm gì, chỉ chờ lệnh mới
    // Không return ngay, vì có thể có should_move từ lệnh vừa nhận
  }
  
  // Thực hiện IK và di chuyển
  if (should_move) {
    float q_rad[6];
    bool valid = false;
    
    // Kiểm tra xem có phải chế độ auto orient không
    bool use_auto_orient = false;
    if (control_mode == MODE_QUEUE && queue_step_active) {
      use_auto_orient = queue_auto_orient[queue_index];
    }
    
    if (use_auto_orient) {
      // Chế độ IK0: Tự động tìm hướng hợp lệ
      float found_rpy[3];
      valid = ik6dof_auto_orient(p06[0], p06[1], p06[2], q_rad, found_rpy);
      
      if (valid) {
        Serial.print(F("  Auto-found orientation: RPY=("));
        Serial.print(found_rpy[0]*180/PI, 2); Serial.print(F(" deg,"));
        Serial.print(found_rpy[1]*180/PI, 2); Serial.print(F(" deg,"));
        Serial.print(found_rpy[2]*180/PI, 2); Serial.println(F(" deg)"));
        
        // Cập nhật RPY cho điểm này
        rpy[0] = found_rpy[0];
        rpy[1] = found_rpy[1];
        rpy[2] = found_rpy[2];
      }
    } else {
      // Chế độ thông thường: Dùng hướng đã cho
      valid = ik6dof_from_p06_rpy(p06[0], p06[1], p06[2], 
                                   rpy[0], rpy[1], rpy[2], 
                                   q_rad);
    }
    
    // Apply J6 extra rotation if in queue mode
    if (valid && control_mode == MODE_QUEUE && queue_step_active) {
      float j6_extra = queue_j6_extra[queue_index];
      if (j6_extra != 0) {
        float before = q_rad[5] * 180 / PI;
        q_rad[5] += j6_extra; // Add extra rotation to Joint 6
        float after = q_rad[5] * 180 / PI;
        Serial.print(F("J6: "));
        Serial.print(before, 1);
        Serial.print(F(" deg -> "));
        Serial.print(after, 1);
        Serial.print(F(" deg (+"));
        Serial.print(j6_extra * 180 / PI, 1);
        Serial.println(F(" deg)"));
      }
    }
    
    if (valid) {
      // In mức IK đã dùng
      switch (g_ik_level) {
        case 1:
          Serial.println(F("[IK] Level 1: J1,J2,J3,J5 du 6DOF - J4=0, J6=0"));
          break;
        case 2:
          Serial.println(F("[IK] Level 2: J1,J2,J3,J4,J5 du 6DOF - J6=0"));
          break;
        default:
          Serial.println(F("[IK] Level 3: Day du 6 khop"));
          break;
      }
    }

    if (!valid) {
      Serial.print(F("ERROR: IK FAILED - Position unreachable! P=("));
      Serial.print(p06[0], 3); Serial.print(F(","));
      Serial.print(p06[1], 3); Serial.print(F(","));
      Serial.print(p06[2], 3); Serial.print(F(") RPY=("));
      Serial.print(rpy[0], 4); Serial.print(F(","));
      Serial.print(rpy[1], 4); Serial.print(F(","));
      Serial.print(rpy[2], 4); Serial.println(F(")rad"));
      if (control_mode == MODE_MANUAL) {
        control_mode = MODE_IDLE;
        Serial.println(F("Switched to IDLE mode"));
      } else if (control_mode == MODE_QUEUE) {
        control_mode = MODE_IDLE;
        queue_running = false;
        queue_step_active = false;
        Serial.println(F("Queue stopped due to IK failure"));
      }
      delay(100);
      return;
    }
    
    // Lưu góc IK (trước bù J4/J5) làm tham chiếu cho lần IK tiếp theo
    // Phải lưu TRƯỚC KHI trừ offset J4/J5
    for (int k = 0; k < 6; k++) q_current_ik[k] = q_rad[k];

    // J4: trừ 180° (PI) khỏi góc IK trước khi tính xung
    float q4_ik_deg = q_rad[3] * 180.0 / PI;
    q_rad[3] -= PI;
    float q4_motor_deg = q_rad[3] * 180.0 / PI;
    Serial.print(F("[J4] IK=")); Serial.print(q4_ik_deg, 2);
    Serial.print(F("deg  motor=")); Serial.print(q4_motor_deg, 2);
    Serial.print(F("deg  steps="));
    Serial.println(-(long)(q_rad[3] / (2*PI) * STEPS_PER_REV[3]));

    // J5: trừ 90° (PI/2) khỏi góc IK trước khi tính xung
    // Đây là bù cơ học: vị trí vật lý 0° của khớp 5 lệch 90° so với DH convention
    float q5_ik_deg = q_rad[4] * 180.0 / PI;          // góc IK gốc (để debug)
    q_rad[4] -= PI / 2.0;                              // trừ 90° → góc động cơ thực tế
    float q5_motor_deg = q_rad[4] * 180.0 / PI;        // góc sau khi trừ (để debug)
    Serial.print(F("[J5] IK=")); Serial.print(q5_ik_deg, 2);
    Serial.print(F("deg  motor=")); Serial.print(q5_motor_deg, 2);
    Serial.print(F("deg  steps="));
    Serial.println((long)(q_rad[4] / (2*PI) * STEPS_PER_REV[4]));

    // Chuyển đổi rad -> steps
    long steps[6];
    rad_to_steps(q_rad, steps);
    align_periodic_joint_targets(steps);
    for (int i = 0; i < 6; i++) q_current_ik[i] = steps_to_ik_angle(i, steps[i]);
    
    // Debug: Print ALL joints target and current
    Serial.println(F("All joints:"));
    bool need_to_move = false;
    for (int i = 0; i < 6; i++) {
      long current = JOINTS[i]->currentPosition();
      long delta = steps[i] - current;
      if (delta != 0) need_to_move = true;
      Serial.print(F("  J"));
      Serial.print(i+1);
      Serial.print(F(": target="));
      Serial.print(steps[i]);
      Serial.print(F(" current="));
      Serial.print(current);
      Serial.print(F(" delta="));
      Serial.println(delta);
    }
    
    // Kiểm tra xem có cần di chuyển không
    if (!need_to_move) {
      Serial.println(F("Already at target position!"));
      if (control_mode == MODE_MANUAL) {
        control_mode = MODE_IDLE;
      } else if (control_mode == MODE_QUEUE && queue_step_active) {
        // Xử lý gripper giống nhánh all_done
        int8_t d3 = queue_d3_state[queue_index];
        float gdelay = queue_gripper_delay_s[queue_index];

        if (d3 != -1) {
          if (gdelay > 0.0f) {
            waiting_for_gripper = true;
            gripper_delay_start = millis();
            Serial.print(F("[GRIPPER] Se kich D3 sau "));
            Serial.print(gdelay, 1);
            Serial.println(F("s..."));
          } else {
            if (d3 == 1) {
              digitalWrite(GRIPPER_PIN, HIGH);
              Serial.println(F("Gripper D3 -> HIGH (dong)"));
            } else {
              digitalWrite(GRIPPER_PIN, LOW);
              Serial.println(F("Gripper D3 -> LOW (mo)"));
            }
          }
        }

        if (!waiting_for_gripper) {
          if (queue_delay_s[queue_index] > 0.0f) {
            waiting_for_delay = true;
            move_delay_start = millis();
          } else {
            queue_index++;
            queue_step_active = false;
            if (queue_index >= queue_count) {
              control_mode = MODE_IDLE;
              queue_running = false;
              Serial.println(F("QUEUE COMPLETED"));
            }
          }
        }
      }
      return;
    }
    
    // In góc khớp (chỉ ở manual mode)
    if (control_mode == MODE_MANUAL) {
      Serial.print(F("Joint angles (deg): "));
      for (int i = 0; i < 6; i++) {
        Serial.print(q_rad[i]*180/PI, 1);
        if (i < 5) Serial.print(F(", "));
      }
      Serial.println();
      Serial.println(F("Moving..."));
    }
    
    // Di chuyển động cơ (non-blocking)
    digitalWrite(13, HIGH);
    beginCoordinatedMove(steps);
    is_moving = true; // Đánh dấu đang di chuyển
    // Planner moi dung moveTo() + run() rieng cho tung khop
    // de dong bo thoi gian va dung em hon tai target
  }
  
  delay(10);
}
