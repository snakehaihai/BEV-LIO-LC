#include <cmath>
#include <math.h>
#include <deque>
#include <mutex>
#include <thread>
#include <fstream>
#include <csignal>
#include <ros/ros.h>
#include <so3_math.h>
#include <Eigen/Eigen>
#include <common_lib.h>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <condition_variable>
#include <nav_msgs/Odometry.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Vector3.h>
#include "use-ikfom.hpp"
#include "preprocess.h"
#include <unordered_map>

/// *************Preconfiguration

#define MAX_INI_COUNT (10)

const bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);};

/// *************IMU Process and undistortion
class ImuProcess
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess();
  ~ImuProcess();
  
  void Reset();
  void Reset(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);
  void set_extrinsic(const V3D &transl, const M3D &rot);
  void set_extrinsic(const V3D &transl);
  void set_extrinsic(const MD(4,4) &T);
  void set_gyr_cov(const V3D &scaler);
  void set_acc_cov(const V3D &scaler);
  void set_gyr_bias_cov(const V3D &b_g);
  void set_acc_bias_cov(const V3D &b_a);
  Eigen::Matrix<double, 12, 12> Q;
  void Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr pcl_un_);
  void Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr cur_pcl_un_, std::vector<M4D>& vec_T_Li_Lk, std::vector<int>& vec_idx);
  void Process(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr cur_pcl_un_,M4D& T_current_Li);
  ofstream fout_imu;
  V3D cov_acc;
  V3D cov_gyr;
  V3D cov_acc_scale;
  V3D cov_gyr_scale;
  V3D cov_bias_gyr;
  V3D cov_bias_acc;
  double first_lidar_time;
  int lidar_type;

 private:
  void IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N);
  void UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_in_out);
  void UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, 
    PointCloudXYZI &pcl_in_out, std::vector<M4D>& vec_T_Li_Lk, std::vector<int>& vec_idx);
  void UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_out, M4D& T_current_Li);

  PointCloudXYZI::Ptr cur_pcl_un_;
  sensor_msgs::ImuConstPtr last_imu_;
  deque<sensor_msgs::ImuConstPtr> v_imu_;
  vector<Pose6D> IMUpose;
  vector<M3D>    v_rot_pcl_;
  M3D Lidar_R_wrt_IMU;
  V3D Lidar_T_wrt_IMU;
  V3D mean_acc;
  V3D mean_gyr;
  V3D angvel_last;
  V3D acc_s_last;
  double start_timestamp_;
  double last_lidar_end_time_;
  int    init_iter_num = 1;
  bool   b_first_frame_ = true;
  bool   imu_need_init_ = true;
};

ImuProcess::ImuProcess()
    : b_first_frame_(true), imu_need_init_(true), start_timestamp_(-1)
{
  init_iter_num = 1;
  Q = process_noise_cov();
  cov_acc       = V3D(0.1, 0.1, 0.1);
  cov_gyr       = V3D(0.1, 0.1, 0.1);
  cov_bias_gyr  = V3D(0.0001, 0.0001, 0.0001);
  cov_bias_acc  = V3D(0.0001, 0.0001, 0.0001);
  mean_acc      = V3D(0, 0, -1.0);
  mean_gyr      = V3D(0, 0, 0);
  angvel_last     = Zero3d;
  Lidar_T_wrt_IMU = Zero3d;
  Lidar_R_wrt_IMU = Eye3d;
  last_imu_.reset(new sensor_msgs::Imu());
}

ImuProcess::~ImuProcess() {}

void ImuProcess::Reset() 
{
  // ROS_WARN("Reset ImuProcess");
  mean_acc      = V3D(0, 0, -1.0);
  mean_gyr      = V3D(0, 0, 0);
  angvel_last       = Zero3d;
  imu_need_init_    = true;
  start_timestamp_  = -1;
  init_iter_num     = 1;
  v_imu_.clear();
  IMUpose.clear();
  last_imu_.reset(new sensor_msgs::Imu());
  cur_pcl_un_.reset(new PointCloudXYZI());
}

void ImuProcess::set_extrinsic(const MD(4,4) &T)
{
  Lidar_T_wrt_IMU = T.block<3,1>(0,3);
  Lidar_R_wrt_IMU = T.block<3,3>(0,0);
}

void ImuProcess::set_extrinsic(const V3D &transl)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU.setIdentity();
}

void ImuProcess::set_extrinsic(const V3D &transl, const M3D &rot)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU = rot;
}

void ImuProcess::set_gyr_cov(const V3D &scaler)
{
  cov_gyr_scale = scaler;
}

void ImuProcess::set_acc_cov(const V3D &scaler)
{
  cov_acc_scale = scaler;
}

void ImuProcess::set_gyr_bias_cov(const V3D &b_g)
{
  cov_bias_gyr = b_g;
}

void ImuProcess::set_acc_bias_cov(const V3D &b_a)
{
  cov_bias_acc = b_a;
}

void ImuProcess::IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N)
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity **/
  
  V3D cur_acc, cur_gyr;
  
  if (b_first_frame_)
  {
    Reset();
    N = 1;
    b_first_frame_ = false;
    const auto &imu_acc = meas.imu.front()->linear_acceleration;
    const auto &gyr_acc = meas.imu.front()->angular_velocity;
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;
    first_lidar_time = meas.lidar_beg_time;
  }

  for (const auto &imu : meas.imu)
  {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    mean_acc      += (cur_acc - mean_acc) / N;
    mean_gyr      += (cur_gyr - mean_gyr) / N;

    cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) / (N * N);
    cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) * (N - 1.0) / (N * N);

    // cout<<"acc norm: "<<cur_acc.norm()<<" "<<mean_acc.norm()<<endl;

    N ++;
  }
  state_ikfom init_state = kf_state.get_x();
  init_state.grav = S2(- mean_acc / mean_acc.norm() * G_m_s2);
  
  //state_inout.rot = Eye3d; // Exp(mean_acc.cross(V3D(0, 0, -1 / scale_gravity)));
  init_state.bg  = mean_gyr;
  init_state.offset_T_L_I = Lidar_T_wrt_IMU;
  init_state.offset_R_L_I = Lidar_R_wrt_IMU;
  kf_state.change_x(init_state);

  esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = kf_state.get_P();
  init_P.setIdentity();
  init_P(6,6) = init_P(7,7) = init_P(8,8) = 0.00001;
  init_P(9,9) = init_P(10,10) = init_P(11,11) = 0.00001;
  init_P(15,15) = init_P(16,16) = init_P(17,17) = 0.0001;
  init_P(18,18) = init_P(19,19) = init_P(20,20) = 0.001;
  init_P(21,21) = init_P(22,22) = 0.00001; 
  kf_state.change_P(init_P);
  last_imu_ = meas.imu.back();

}

void ImuProcess::UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_out)
{
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu_);
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();

  double pcl_beg_time = meas.lidar_beg_time;
  double pcl_end_time = meas.lidar_end_time;

    if (lidar_type == MARSIM) {
        pcl_beg_time = last_lidar_end_time_;
        pcl_end_time = meas.lidar_beg_time;
    }

    /*** sort point clouds by offset time ***/
  pcl_out = *(meas.lidar);
  sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
  // cout<<"[ IMU Process ]: Process lidar from "<<pcl_beg_time<<" to "<<pcl_end_time<<", " \
  //          <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to "<<imu_end_time<<endl;

  /*** Initialize IMU pose ***/
  state_ikfom imu_state = kf_state.get_x();
  IMUpose.clear();
  IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));

  /*** forward propagation at each imu point ***/
  V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
  M3D R_imu;

  double dt = 0;

  input_ikfom in;
  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
  {
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);
    
    if (tail->header.stamp.toSec() < last_lidar_end_time_)    continue;
    
    angvel_avr<<0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr   <<0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    // fout_imu << setw(10) << head->header.stamp.toSec() - first_lidar_time << " " << angvel_avr.transpose() << " " << acc_avr.transpose() << endl;

    acc_avr     = acc_avr * G_m_s2 / mean_acc.norm(); // - state_inout.ba;

    if(head->header.stamp.toSec() < last_lidar_end_time_)
    {
      dt = tail->header.stamp.toSec() - last_lidar_end_time_;
      // dt = tail->header.stamp.toSec() - pcl_beg_time;
    }
    else
    {
      dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
    }
    
    in.acc = acc_avr;
    in.gyro = angvel_avr;
    Q.block<3, 3>(0, 0).diagonal() = cov_gyr;
    Q.block<3, 3>(3, 3).diagonal() = cov_acc;
    Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
    Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;
    kf_state.predict(dt, Q, in);

    /* save the poses at each IMU measurements */
    imu_state = kf_state.get_x();
    angvel_last = angvel_avr - imu_state.bg;
    acc_s_last  = imu_state.rot * (acc_avr - imu_state.ba);
    for(int i=0; i<3; i++)
    {
      acc_s_last[i] += imu_state.grav[i];
    }
    double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
    IMUpose.push_back(set_pose6d(offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));
  }

  /*** calculated the pos and attitude prediction at the frame-end ***/
  double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
  dt = note * (pcl_end_time - imu_end_time);
  kf_state.predict(dt, Q, in);
  
  imu_state = kf_state.get_x();
  last_imu_ = meas.imu.back();
  last_lidar_end_time_ = pcl_end_time;

  /*** undistort each lidar point (backward propagation) ***/
  if (pcl_out.points.begin() == pcl_out.points.end()) return;

  if(lidar_type != MARSIM){
      auto it_pcl = pcl_out.points.end() - 1;
      for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
      {
          auto head = it_kp - 1;
          auto tail = it_kp;
          R_imu<<MAT_FROM_ARRAY(head->rot);
          // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
          vel_imu<<VEC_FROM_ARRAY(head->vel);
          pos_imu<<VEC_FROM_ARRAY(head->pos);
          acc_imu<<VEC_FROM_ARRAY(tail->acc);
          angvel_avr<<VEC_FROM_ARRAY(tail->gyr);

          for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl --)
          {
              dt = it_pcl->curvature / double(1000) - head->offset_time;

              /* Transform to the 'end' frame, using only the rotation
               * Note: Compensation direction is INVERSE of Frame's moving direction
               * So if we want to compensate a point at timestamp-i to the frame-e
               * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
              M3D R_i(R_imu * Exp(angvel_avr, dt));

              V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
              V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);
              V3D P_compensate = imu_state.offset_R_L_I.conjugate() * (imu_state.rot.conjugate() * (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);// not accurate!

              // save Undistorted points and their rotation
              it_pcl->x = P_compensate(0);
              it_pcl->y = P_compensate(1);
              it_pcl->z = P_compensate(2);

              if (it_pcl == pcl_out.points.begin()) break;
          }
      }
  }
}

void ImuProcess::Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr cur_pcl_un_)
{
  double t1,t2,t3;
  t1 = omp_get_wtime();

  if(meas.imu.empty()) {return;};
  ROS_ASSERT(meas.lidar != nullptr);

  if (imu_need_init_)
  {
    /// The very first lidar frame
    IMU_init(meas, kf_state, init_iter_num);

    imu_need_init_ = true;
    
    last_imu_   = meas.imu.back();

    state_ikfom imu_state = kf_state.get_x();
    if (init_iter_num > MAX_INI_COUNT)
    {
      cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
      imu_need_init_ = false;

      cov_acc = cov_acc_scale;
      cov_gyr = cov_gyr_scale;
      ROS_INFO("IMU Initial Done");
      // ROS_INFO("IMU Initial Done: Gravity: %.4f %.4f %.4f %.4f; state.bias_g: %.4f %.4f %.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f",\
      //          imu_state.grav[0], imu_state.grav[1], imu_state.grav[2], mean_acc.norm(), cov_bias_gyr[0], cov_bias_gyr[1], cov_bias_gyr[2], cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1], cov_gyr[2]);
      fout_imu.open(DEBUG_FILE_DIR("imu.txt"),ios::out);
    }

    return;
  }

  UndistortPcl(meas, kf_state, *cur_pcl_un_);

  t2 = omp_get_wtime();
  t3 = omp_get_wtime();
  
  // cout<<"[ IMU Process ]: Time: "<<t3 - t1<<endl;
}
// vec_T_Li_Lk 从LiDAR参考坐标系（在时间戳Li处）到当前坐标系（Lk）的变换矩阵的向量
// vec_idx 
void ImuProcess::UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, 
  PointCloudXYZI &pcl_out, std::vector<M4D>& vec_T_Li_Lk,
  std::vector<int>& vec_idx)
{
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu_);
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();
  const double &pcl_beg_time = meas.lidar_beg_time;
  const double &pcl_end_time = meas.lidar_end_time;

  /*** store points at equal timestamps and the corresponding point indices ***/
  std::unordered_map<int, std::vector<int>> timestamp_map;
  timestamp_map.reserve(pcl_out.points.size());
  for (size_t idx = 0; idx < pcl_out.points.size(); idx++)
  {
    const int t_ns = pcl_out.points[idx].curvature * 1e6;
    timestamp_map[t_ns].push_back(idx);
  }

  /*** extract individual timestamps and sort times in ascending order ***/
  std::vector<int> timestamps;
  timestamps.reserve(timestamp_map.size());
  for (auto it = timestamp_map.begin(); it != timestamp_map.end(); it++)
  {
    timestamps.push_back(it->first);
  }
  std::sort(timestamps.begin(), timestamps.end());

  /*** initialize IMU pose ***/
  state_ikfom imu_state = kf_state.get_x();
  IMUpose.clear();
  IMUpose.reserve(v_imu.size());
  IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, 
    imu_state.rot.toRotationMatrix()));

  /*** forward propagation at each imu point ***/
  V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
  M3D R_imu;

  double dt = 0;

  input_ikfom in;
  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
  {
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);

    if (tail->header.stamp.toSec() < last_lidar_end_time_)    continue;

    angvel_avr<<0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr   <<0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    acc_avr     = acc_avr * G_m_s2 / mean_acc.norm(); // - state_inout.ba;

    if(head->header.stamp.toSec() < last_lidar_end_time_)
    {
      dt = tail->header.stamp.toSec() - last_lidar_end_time_;
    }
    else
    {
      dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
    }

    in.acc = acc_avr;
    in.gyro = angvel_avr;
    Q.block<3, 3>(0, 0).diagonal() = cov_gyr;
    Q.block<3, 3>(3, 3).diagonal() = cov_acc;
    Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
    Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;
    kf_state.predict(dt, Q, in);

    /* save the poses at each IMU measurements */
    imu_state = kf_state.get_x();
    angvel_last = angvel_avr - imu_state.bg;
    acc_s_last  = imu_state.rot * (acc_avr - imu_state.ba);
    for(int i=0; i<3; i++)
    {
      acc_s_last[i] += imu_state.grav[i];
    }
    double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
    IMUpose.push_back(set_pose6d(offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, 
      imu_state.rot.toRotationMatrix()));
  }

  /*** calculated the pos and attitude prediction at the frame-end ***/
  double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
  dt = note * (pcl_end_time - imu_end_time);
  kf_state.predict(dt, Q, in);

  imu_state = kf_state.get_x();
  last_imu_ = meas.imu.back();
  last_lidar_end_time_ = pcl_end_time;

  if (pcl_out.points.begin() == pcl_out.points.end()) return;

  /*** pose at frame-end ***/
  M4D T_Ik_G = M4D::Identity();
  T_Ik_G.block<3,3>(0,0) = imu_state.rot.toRotationMatrix().transpose();
  T_Ik_G.block<3,1>(0,3) = -T_Ik_G.block<3,3>(0,0) * imu_state.pos;
  /*** transform from Lidar to IMU ***/
  M4D T_I_L = M4D::Identity();
  T_I_L.block<3, 3>(0, 0) = imu_state.offset_R_L_I.toRotationMatrix();
  T_I_L.block<3, 1>(0, 3) = imu_state.offset_T_L_I;
  /*** transform from IMU to Lidar ***/
  M4D T_L_I = M4D::Identity();
  T_L_I.block<3,3>(0,0) = T_I_L.block<3,3>(0,0).transpose();
  T_L_I.block<3,1>(0,3) = -T_I_L.block<3,3>(0,0).transpose() * 
    T_I_L.block<3,1>(0,3);

  M4D T_G_Ii = M4D::Identity();
  vec_T_Li_Lk = std::vector<M4D>(timestamps.size(), M4D::Identity());
  std::vector<M4D> vec_T_Lk_Li(timestamps.size(), M4D::Identity());

  int t_idx = timestamps.size() - 1;

  /*** calculate T_Lk_Li at required times ***/
  for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
  {
    auto head = it_kp - 1;
    auto tail = it_kp;
    R_imu<<MAT_FROM_ARRAY(head->rot);
    vel_imu<<VEC_FROM_ARRAY(head->vel);
    pos_imu<<VEC_FROM_ARRAY(head->pos);
    acc_imu<<VEC_FROM_ARRAY(tail->acc);
    angvel_avr<<VEC_FROM_ARRAY(tail->gyr);
    
    for(; float(timestamps[t_idx])*1.e-9 > head->offset_time; t_idx--)
    {
      dt = float(timestamps[t_idx])*1.e-9 - head->offset_time;

      /*** IMU pose at time i ***/
      T_G_Ii.block<3, 3>(0, 0) = R_imu * Exp(angvel_avr, dt);
      T_G_Ii.block<3, 1>(0, 3) = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;

      /*** T_Lk_Li ***/
      vec_T_Lk_Li[t_idx] = T_L_I * (T_Ik_G * T_G_Ii * T_I_L);

      /*** T_Li_Lk ***/
      vec_T_Li_Lk[t_idx].block<3,3>(0,0) = 
          vec_T_Lk_Li[t_idx].block<3,3>(0,0).transpose();
      vec_T_Li_Lk[t_idx].block<3,1>(0,3) = 
        - vec_T_Li_Lk[t_idx].block<3,3>(0,0) * vec_T_Lk_Li[t_idx].block<3,1>(0,3);
  
      if (t_idx == 0) break;
    }
    if (t_idx == 0) break;
  }

  /*** undistort the lidar points using the respective T_LK_Li ***/
  #ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
    #pragma omp parallel for
  #endif
  for (size_t idx = 0u; idx < timestamps.size(); idx++) {
    for (int p_idx : timestamp_map[timestamps[idx]]) {
      V3D P_i = pcl_out.points[p_idx].getVector3fMap().cast<double>();
      V3D P_k = (vec_T_Lk_Li[idx].block<3,3>(0,0) * P_i) + vec_T_Lk_Li[idx].block<3,1>(0,3);

      pcl_out.points[p_idx].x = P_k(0);
      pcl_out.points[p_idx].y = P_k(1);
      pcl_out.points[p_idx].z = P_k(2);
      vec_idx[p_idx] = idx;
    }
  }
}

void ImuProcess::Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, 
  PointCloudXYZI::Ptr cur_pcl_un_, std::vector<M4D>& vec_T_Li_Lk, std::vector<int>& vec_idx)
{
  if(meas.imu.empty()) {return;};
  ROS_ASSERT(meas.lidar != nullptr);

  if (imu_need_init_)
  {
    /// The very first lidar frame
    IMU_init(meas, kf_state, init_iter_num);

    imu_need_init_ = true;

    last_imu_   = meas.imu.back();

    state_ikfom imu_state = kf_state.get_x();
    if (init_iter_num > MAX_INI_COUNT)
    {
      cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
      imu_need_init_ = false;

      cov_acc = cov_acc_scale;
      cov_gyr = cov_gyr_scale;
      ROS_INFO("IMU Initial Done");
    }
    cur_pcl_un_->points.clear();
    return;
  }

  UndistortPcl(meas, kf_state, *cur_pcl_un_, vec_T_Li_Lk, vec_idx);
}

void ImuProcess::UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_out, M4D& T_current_Li)
{
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu_);
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();
  double pcl_beg_time = meas.lidar_beg_time;
  double pcl_end_time = meas.lidar_end_time;

    if (lidar_type == MARSIM) {
        pcl_beg_time = last_lidar_end_time_;
        pcl_end_time = meas.lidar_beg_time;
    }

    /*** sort point clouds by offset time ***/
  pcl_out = *(meas.lidar);
  sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
  // cout<<"[ IMU Process ]: Process lidar from "<<pcl_beg_time<<" to "<<pcl_end_time<<", " \
  //          <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to "<<imu_end_time<<endl;


  /*** initialize IMU state at the start (Li) ***/
  state_ikfom imu_state_Li = kf_state.get_x();
  IMUpose.clear();
  // IMUpose.reserve(v_imu.size());
  IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, imu_state_Li.vel, imu_state_Li.pos, imu_state_Li.rot.toRotationMatrix()));

  /*** forward propagate IMU to current time ***/
  V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
  M3D R_imu;
  double dt = 0;
  input_ikfom in;
  
  // Process each IMU measurement
  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++) {
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);
    if (tail->header.stamp.toSec() < last_lidar_end_time_) continue;

    // Compute average angular velocity and acceleration
    angvel_avr << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                  0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                  0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr << 0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
               0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
               0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);
    acc_avr = acc_avr * G_m_s2 / mean_acc.norm(); // Normalize acceleration

    // Calculate time difference
    // dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
    if(head->header.stamp.toSec() < last_lidar_end_time_)
    {
      dt = tail->header.stamp.toSec() - last_lidar_end_time_;
      // dt = tail->header.stamp.toSec() - pcl_beg_time;
    }
    else
    {
      dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
    }
    // Perform IMU prediction
    in.acc = acc_avr;
    in.gyro = angvel_avr;
    Q.block<3, 3>(0, 0).diagonal() = cov_gyr;
    Q.block<3, 3>(3, 3).diagonal() = cov_acc;
    Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
    Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;

    kf_state.predict(dt, Q, in);
    
    /* save the poses at each IMU measurements */
    state_ikfom imu_state = kf_state.get_x();
    angvel_last = angvel_avr - imu_state.bg;
    acc_s_last  = imu_state.rot * (acc_avr - imu_state.ba);
    for(int i=0; i<3; i++)
    {
      acc_s_last[i] += imu_state.grav[i];
    }
    double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
    IMUpose.push_back(set_pose6d(offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));
  }

  // IMU state at current time
  // --------计算当前时刻的IMU状态---------------
  state_ikfom imu_state_current = kf_state.get_x();
  last_imu_ = meas.imu.back();
  last_lidar_end_time_ = pcl_end_time;

  // Transformation matrix at time Li
  M4D T_G_I_Li = M4D::Identity(); // Transformation matrix from IMU to global frame at time Li
  T_G_I_Li.block<3,3>(0,0) = imu_state_Li.rot.toRotationMatrix();
  T_G_I_Li.block<3,1>(0,3) = imu_state_Li.pos;

  // Transformation matrix at current time
  M4D T_G_I_current = M4D::Identity();
  T_G_I_current.block<3,3>(0,0) = imu_state_current.rot.toRotationMatrix();
  T_G_I_current.block<3,1>(0,3) = imu_state_current.pos;

  /*** Calculate T_current_Li transformation ***/
  M4D T_L_I = M4D::Identity();
  T_L_I.block<3, 3>(0, 0) = imu_state_Li.offset_R_L_I.toRotationMatrix(); // IMU to Lidar
  T_L_I.block<3, 1>(0, 3) = imu_state_Li.offset_T_L_I;
  // 计算方式： 先从I_Li到Global，在从Global到I_current
  // ，最后从I_current到Lidar，得到的是从Li到当前时刻的变换矩阵，坐标系是Lidar
  T_current_Li.block<3,3>(0,0) = T_L_I.block<3,3>(0,0) * (T_G_I_current.block<3,3>(0,0) * T_G_I_Li.block<3,3>(0,0).inverse());
  T_current_Li.block<3,1>(0,3) = T_L_I.block<3,3>(0,0) * (T_G_I_current.block<3,1>(0,3) - T_G_I_Li.block<3,1>(0,3));
  T_current_Li = T_L_I * (T_G_I_current * T_G_I_Li.inverse()); // 从Li到当前时刻的变换矩阵，使用时：P_current = T_current_Li * P_Li
// -------------------------------------------------------

  /*** calculated the pos and attitude prediction at the frame-end ***/
  double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
  dt = note * (pcl_end_time - imu_end_time);
  kf_state.predict(dt, Q, in);
  
  state_ikfom imu_state = kf_state.get_x();
  last_imu_ = meas.imu.back();
  last_lidar_end_time_ = pcl_end_time;

  /*** undistort each lidar point (backward propagation) ***/
  if (pcl_out.points.begin() == pcl_out.points.end()) return;

  if(lidar_type != MARSIM){
      auto it_pcl = pcl_out.points.end() - 1;
      for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
      {
          auto head = it_kp - 1;
          auto tail = it_kp;
          R_imu<<MAT_FROM_ARRAY(head->rot);
          // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
          vel_imu<<VEC_FROM_ARRAY(head->vel);
          pos_imu<<VEC_FROM_ARRAY(head->pos);
          acc_imu<<VEC_FROM_ARRAY(tail->acc);
          angvel_avr<<VEC_FROM_ARRAY(tail->gyr);

          for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl --)
          {
              dt = it_pcl->curvature / double(1000) - head->offset_time;

              /* Transform to the 'end' frame, using only the rotation
               * Note: Compensation direction is INVERSE of Frame's moving direction
               * So if we want to compensate a point at timestamp-i to the frame-e
               * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
              M3D R_i(R_imu * Exp(angvel_avr, dt));

              V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
              V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);
              V3D P_compensate = imu_state.offset_R_L_I.conjugate() * (imu_state.rot.conjugate() * (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);// not accurate!

              // save Undistorted points and their rotation
              it_pcl->x = P_compensate(0);
              it_pcl->y = P_compensate(1);
              it_pcl->z = P_compensate(2);

              if (it_pcl == pcl_out.points.begin()) break;
          }
      }
  }
}

void ImuProcess::Process(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr cur_pcl_un_,M4D& T_current_Li)
{
  if (meas.imu.empty()) { return; };
  ROS_ASSERT(meas.lidar != nullptr);

  if (imu_need_init_)
  {
    /// The very first lidar frame
    IMU_init(meas, kf_state, init_iter_num);
    imu_need_init_ = true;
    last_imu_ = meas.imu.back();
    state_ikfom imu_state = kf_state.get_x();
    
    if (init_iter_num > MAX_INI_COUNT)
    {
      cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
      imu_need_init_ = false;
      cov_acc = cov_acc_scale;
      cov_gyr = cov_gyr_scale;
      ROS_INFO("IMU Initial Done");
      // ROS_INFO("IMU Initial Done: Gravity: %.4f %.4f %.4f %.4f; state.bias_g: %.4f %.4f %.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f",\
      //          imu_state.grav[0], imu_state.grav[1], imu_state.grav[2], mean_acc.norm(), cov_bias_gyr[0], cov_bias_gyr[1], cov_bias_gyr[2], cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1], cov_gyr[2]);
      fout_imu.open(DEBUG_FILE_DIR("imu.txt"),ios::out);
    }
    return;
  }

  // Call simplified UndistortPcl function
  UndistortPcl(meas, kf_state, *cur_pcl_un_,T_current_Li);
}

