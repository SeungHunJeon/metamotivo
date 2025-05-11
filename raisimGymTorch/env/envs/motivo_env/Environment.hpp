//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <stdlib.h>
#include <set>
#include "../../RaisimGymEnv.hpp"


namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {

 public:

  explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {
    /// create world
    world_ = std::make_unique<raisim::World>();

    /// add objects
    robot_ = world_->addArticulatedSystem(resourceDir_+"/humanoid.urdf");
    robot_->setName("robot");
    robot_->setControlMode(raisim::ControlMode::FORCE_AND_TORQUE);
    world_->addGround();

    /// get robot data
    gcDim_ = robot_->getGeneralizedCoordinateDim();
    gvDim_ = robot_->getDOF();
    nJoints_ = gvDim_ - 6;

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nJoints_);

    /// this is nominal configuration of anymal
    gc_init_.head(7) << 0, 0, 1.0, 0.70710678, 0.70710678, 0, 0;

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(50.0);
    jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(0.2);
    // robot_->setPdGains(jointPgain, jointDgain);
    robot_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 358;
    actionDim_ = nJoints_; actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
    obDouble_.setZero(obDim_);

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    double action_std;
    READ_YAML(double, action_std, cfg_["action_std"]) /// example of reading params from the config
    actionStd_.setConstant(action_std);

    /// Reward coefficients
    rewards_.initializeFromConfigurationFile (cfg["reward"]);

    /// indices of links that should not make contact with ground
    footIndices_.insert(robot_->getBodyIdx("LF_SHANK"));
    footIndices_.insert(robot_->getBodyIdx("RF_SHANK"));
    footIndices_.insert(robot_->getBodyIdx("LH_SHANK"));
    footIndices_.insert(robot_->getBodyIdx("RH_SHANK"));

    JointLimits = robot_->getJointLimits();

    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer();
      server_->focusOn(robot_);
    }
    
  }

  void setSeed(int seed) {
    gen_.seed(seed);
  }

  void init() final { }

  void generateRandomAction(Eigen::VectorXf& action, int nJoints) {
    for (int i = 0; i < nJoints; i++) {
      action(i) = (uniDist_(gen_) - 0.5) * 1.0; // [-0.5, 0.5)
    }
  }


  void reset_fall() {
    Eigen::VectorXd gc_init = gc_init_;
    Eigen::VectorXd gv_init = gv_init_;

    double theta = uniDist_(gen_) * 2 * M_PI;
    raisim::Vec<3> angle_axis;
    raisim::Vec<4> quat;

    gc_init[2] = 1.0;
    angle_axis[0] = (uniDist_(gen_) - 0.5) * 2;
    angle_axis[1] = (uniDist_(gen_) - 0.5) * 2;
    angle_axis[2] = (uniDist_(gen_) - 0.5) * 2;
    angle_axis /= angle_axis.norm();
    raisim::angleAxisToQuaternion(angle_axis, theta, quat);
    gc_init.segment(3, 4) << quat.e();

    robot_->setGeneralizedCoordinate(gc_init_);
    robot_->setGeneralizedVelocity(gv_init_);

    int n_steps = uniIntDist_(gen_);
    Eigen::VectorXf action = Eigen::VectorXf::Zero(nJoints_);
//    double real_time = world_->getWorldTime();
    for (int i = 0; i < n_steps; i++) {
      generateRandomAction(action, nJoints_);
      step(action);
    }
//    world_->setWorldTime(real_time);
  }

  void reset(const Eigen::Ref<EigenVec>& gc_init, const Eigen::Ref<EigenVec>& gv_init) {
    RSINFO("Resetting the environment with random joint angles and velocities.")
    robot_->setState(gc_init.cast<double>(), gv_init.cast<double>());

    if(uniDist_(gen_) < 0.2) {
      RSINFO("Falling down the robot.")
      reset_fall();
    }

    stepCounter_ = 0;
    RSINFO("Resetting done")
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    /// action scaling
    Eigen::VectorXd action_mujoco = action.cast<double>();
    Eigen::VectorXd action_raisim(nJoints_);
    for (int i=0; i<69; i++) {
      if (action_mujoco[i] < -1)
        action_mujoco[i] = -1;
      else if (action_mujoco[i] > 1)
        action_mujoco[i] = 1;
      action_raisim[i] = gainprm[i] * action_mujoco[i] + biasprm[i][0] + biasprm[i][1] * q[i] + biasprm[i][2] * q_dot[i];
      if (action_raisim[i] < forcerange[i][0])
        action_raisim[i] = forcerange[i][0];
      else if (action_raisim[i] > forcerange[i][1])
        action_raisim[i] = forcerange[i][1];
    }
    Eigen::VectorXd GF;
    GF.setZero(gvDim_);
    GF.tail(nJoints_) = action_raisim;

    robot_->setGeneralizedForce(GF);

    for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-10); i++){
      if(server_) server_->lockVisualizationServerMutex();
      world_->integrate();
      if(server_) server_->unlockVisualizationServerMutex();
    }
    stepCounter_++;
    // std::cout << action_raisim[0] << std::endl;

    return control_dt_;
  }

  void updateStateVariable() {
    q = robot_->getGeneralizedCoordinate().e().tail(nJoints_);
    q_dot = robot_->getGeneralizedVelocity().e().tail(nJoints_);
  }

  void updateObservation() {
    updateStateVariable();

    std::vector<Eigen::Vector3d> body_pos;
    for (int i = 0; i < FrameName.size(); i++) {
      raisim::Vec<3> temp;
      robot_->getFramePosition(FrameName[i], temp);
      Eigen::Vector3d temp_eigen(temp[0], temp[1], temp[2]);
      body_pos.push_back(temp_eigen);
    }

    std::vector<Eigen::Quaterniond> body_rot;
    for (int i = 0; i < FrameName.size(); i++) {
      raisim::Mat<3,3> rotmat;
      raisim::Vec<4> temp;
      robot_->getFrameOrientation(FrameName[i], rotmat);
      raisim::rotMatToQuat(rotmat, temp);
      Eigen::Quaterniond temp_eigen(temp[0], temp[1], temp[2], temp[3]);
      body_rot.push_back(temp_eigen);
    }

    std::vector<Eigen::Vector3d> body_vel;
    for (int i = 0; i < FrameName.size(); i++) {
      raisim::Vec<3> temp;
      robot_->getFrameVelocity(FrameName[i], temp);
      Eigen::Vector3d temp_eigen(temp[0], temp[1], temp[2]);
      body_vel.push_back(temp_eigen);
    }

    std::vector<Eigen::Vector3d> body_anv;
    for (int i = 0; i < FrameName.size(); i++) {
      raisim::Vec<3> temp;
      robot_->getFrameAngularVelocity(FrameName[i], temp);
      Eigen::Vector3d temp_eigen(temp[0], temp[1], temp[2]);
      body_anv.push_back(temp_eigen);
    }

    Eigen::Vector3d root_pos = body_pos[0];
    Eigen::Quaterniond remove_base_rot(0.5, -0.5, -0.5, -0.5);
    Eigen::Quaterniond root_rot = body_rot[0] * remove_base_rot;
    Eigen::Vector3d ref_dir(1.0, 0.0, 0.0);
    Eigen::Vector3d rot_dir = root_rot * ref_dir;
    double heading = std::atan2(rot_dir[1], rot_dir[0]);
    Eigen::Quaterniond heading_rot_inv(std::cos(-heading/2), 0, 0, std::sin(-heading/2));

    obDouble_[0] = body_pos[0][2];
    for (int i = 1; i < FrameName.size(); i++) {
      Eigen::Vector3d local_body_pos = heading_rot_inv*(body_pos[i]-root_pos);
      obDouble_[3*i-2] = local_body_pos[0];
      obDouble_[3*i-1] = local_body_pos[1];
      obDouble_[3*i] = local_body_pos[2];
    }
    for (int i = 0; i < FrameName.size(); i++) {
      Eigen::Quaterniond local_body_quat = heading_rot_inv * body_rot[i];
      Eigen::Vector3d ref_tan(1, 0, 0);
      Eigen::Vector3d ref_norm(0, 0, 1);
      Eigen::VectorXd local_body_rot(6);
      local_body_rot.head(3) = local_body_quat * ref_tan;
      local_body_rot.tail(3) = local_body_quat * ref_norm;
      obDouble_[6*i + 70] = local_body_rot[0];
      obDouble_[6*i + 70 +1] = local_body_rot[1];
      obDouble_[6*i + 70 +2] = local_body_rot[2];
      obDouble_[6*i + 70 +3] = local_body_rot[3];
      obDouble_[6*i + 70 +4] = local_body_rot[4];
      obDouble_[6*i + 70 +5] = local_body_rot[5];
    }
    for (int i = 0; i < FrameName.size(); i++) {
      Eigen::Vector3d local_body_vel = heading_rot_inv * body_vel[i];
      obDouble_[3*i + 214] = local_body_vel[0];
      obDouble_[3*i + 214 +1] = local_body_vel[1];
      obDouble_[3*i + 214 +2] = local_body_vel[2];
    }
    for (int i = 0; i < FrameName.size(); i++) {
      Eigen::Vector3d local_body_anv = heading_rot_inv * body_anv[i];
      obDouble_[3*i + 286] = local_body_anv[0];
      obDouble_[3*i + 286 +1] = local_body_anv[1];
      obDouble_[3*i + 286 +2] = local_body_anv[2];
    }
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    updateObservation();
    /// convert it to float
    ob = obDouble_.cast<float>();
  }

  void observeGc(Eigen::Ref<EigenVec> gc) {
    gc = robot_->getGeneralizedCoordinate().e().cast<float>();
  }

  void observeGv(Eigen::Ref<EigenVec> gv) {
    gv = robot_->getGeneralizedVelocity().e().cast<float>();
  }

  void observeStepCounter(int &step_counter_ob) {
    step_counter_ob = stepCounter_;
  }

  bool isTerminalState(float& terminalReward) final {
    terminalReward = float(terminalRewardCoeff_);

    terminalReward = 0.f;
    return false;
  }

  void curriculumUpdate() { };


 private:
  int gcDim_, gvDim_, nJoints_;
  int stepCounter_ = 0;
  bool visualizable_ = false;
  raisim::ArticulatedSystem* robot_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
  double terminalRewardCoeff_ = -10.;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  std::set<size_t> footIndices_;

  std::vector<std::string> FrameName = {
    "floating_base",  // "Pelvis"
    "jointfix_3_4",   // "L_Hip"
    "jointfix_2_8",   // "L_Knee"
    "jointfix_1_12",  // "L_Ankle"
    "jointfix_0_16",  // "L_Toe"
    "jointfix_7_20",  // "R_Hip"
    "jointfix_6_24",  // "R_Knee"
    "jointfix_5_28",  // "R_Ankle"
    "jointfix_4_32",  // "R_Toe"
    "jointfix_22_36", // "Torso"
    "jointfix_21_40", // "Spine"
    "jointfix_20_44", // "Chest"
    "jointfix_9_48",  // "Neck"
    "jointfix_8_52",  // "Head"
    "jointfix_14_56", // "L_Thorax"
    "jointfix_13_60", // "L_Shoulder"
    "jointfix_12_64", // "L_Elbow"
    "jointfix_11_68", // "L_Wrist"
    "jointfix_10_72", // "L_Hand"
    "jointfix_19_76", // "R_Thorax"
    "jointfix_18_80",  // "R_Shoulder"
    "jointfix_17_84", // "R_Elbow"
    "jointfix_16_88", // "R_Wrist"
    "jointfix_15_92"  // "R_Hand"
};

std::vector<double> gainprm = {
  287.33006409732246, 219.9114857512855, 186.3749841742145, 278.10948965903646, 18.001325905069514,
  18.001325905069514, 72.00530362027807, 76.49778111491148, 7.1942471767206255, 71.99902043497087,
  5.755397741376501, 5.755397741376501, 287.33006409732246, 219.9114857512855, 186.3749841742145,
  278.10948965903646, 18.001325905069514, 18.001325905069514, 72.00530362027807, 76.49778111491148,
  7.1942471767206255, 71.99902043497087, 5.755397741376501, 5.755397741376501, 174.00234510682668,
  240.0176787342602, 144.01060724055614, 174.00234510682668, 240.0176787342602, 144.01060724055614,
  174.00234510682668, 240.0176787342602, 144.01060724055614, 27.00198885760427, 25.20185626709732,
  25.20185626709732, 18.001325905069514, 16.801237511398213, 16.801237511398213, 16.200146117011364,
  48.00353574685204, 75.00028861670033, 81.5976331892389, 79.19745640189629, 76.79727961455369,
  13.67849441372996, 123.89813107227425, 66.23733950828719, 21.24973270888136, 3.6002651810139037,
  4.800353574685205, 3.6002651810139037, 10.800795543041708, 19.199319903638422, 16.200146117011364,
  48.00353574685204, 75.00028861670033, 81.5976331892389, 79.19745640189629, 76.79727961455369,
  13.67849441372996, 123.89813107227425, 66.23733950828719, 21.24973270888136, 3.6002651810139037,
  4.800353574685205, 3.6002651810139037, 10.800795543041708, 19.199319903638422
};

std::vector<std::vector<double>> biasprm = {
  {-215.32476047704441, -180, -2},
  {0, -180, -2},
  {96.3683546488669, -180, -2},
  {261.8988715665131, -180, -2},
  {3.612831551628262, -180, -2},
  {-3.612831551628262, -180, -2},
  {0, -90, -2},
  {-40.49512930477244, -90, -2},
  {0, -90, -2},
  {-43.19689898685965, -72, -0.8},
  {0, -72, -0.8},
  {0, -72, -0.8},
  {-215.32476047704441, -180, -2},
  {0, -180, -2},
  {-96.3683546488669, -180, -2},
  {261.8988715665131, -180, -2},
  {-3.612831551628262, -180, -2},
  {3.612831551628262, -180, -2},
  {0, -90, -2},
  {40.49512930477244, -90, -2},
  {0, -90, -2},
  {-43.19689898685965, -72, -0.8},
  {0, -72, -0.8},
  {0, -72, -0.8},
  {66.0153336274335, -240, -4},
  {0, -240, -4},
  {0, -240, -4},
  {66.0153336274335, -240, -4},
  {0, -240, -4},
  {0, -240, -4},
  {66.0153336274335, -240, -4},
  {0, -240, -4},
  {0, -240, -4},
  {9.000662952534757, -36, -0.8},
  {0, -36, -0.8},
  {0, -36, -0.8},
  {6.000441968356505, -24, -0.8},
  {0, -24, -0.8},
  {0, -24, -0.8},
  {-6.607816548050534, -120, -1.6},
  {-12.00088393671301, -120, -1.6},
  {9.00589894029074, -120, -1.6},
  {0, -48, -1.6},
  {-12.00088393671301, -48, -1.6},
  {-4.800353574685214, -48, -1.6},
  {-7.923096672353461, -72, -0.8},
  {-102.29653998619085, -72, -0.8},
  {-60.4819417669107, -72, -0.8},
  {-16.449379134196157, -12, -0.8},
  {0, -12, -0.8},
  {0, -12, -0.8},
  {0, -12, -0.8},
  {0, -12, -0.8},
  {0, -12, -0.8},
  {-6.607816548050534, -120, -1.6},
  {12.000883936713002, -120, -1.6},
  {-9.00589894029074, -120, -1.6},
  {0, -48, -1.6},
  {12.00088393671301, -48, -1.6},
  {4.800353574685204, -48, -1.6},
  {-7.923096672353461, -72, -0.8},
  {102.29653998619085, -72, -0.8},
  {60.481941766910694, -72, -0.8},
  {-16.449379134196157, -12, -0.8},
  {0, -12, -0.8},
  {0, -12, -0.8},
  {0, -12, -0.8},
  {0, -12, -0.8},
  {0, -12, -0.8}
};

std::vector<std::vector<double>> forcerange = {
  {-360.0, 360.0},
  {-360.0, 360.0},
  {-360.0, 360.0},
  {-360.0, 360.0},
  {-360.0, 360.0},
  {-360.0, 360.0},
  {-180.0, 180.0},
  {-180.0, 180.0},
  {-180.0, 180.0},
  {-144.0, 144.0},
  {-144.0, 144.0},
  {-144.0, 144.0},
  {-360.0, 360.0},
  {-360.0, 360.0},
  {-360.0, 360.0},
  {-360.0, 360.0},
  {-360.0, 360.0},
  {-360.0, 360.0},
  {-180.0, 180.0},
  {-180.0, 180.0},
  {-180.0, 180.0},
  {-144.0, 144.0},
  {-144.0, 144.0},
  {-144.0, 144.0},
  {-360.0, 360.0},
  {-360.0, 360.0},
  {-360.0, 360.0},
  {-360.0, 360.0},
  {-360.0, 360.0},
  {-360.0, 360.0},
  {-360.0, 360.0},
  {-360.0, 360.0},
  {-360.0, 360.0},
  {-72.0, 72.0},
  {-72.0, 72.0},
  {-72.0, 72.0},
  {-48.0, 48.0},
  {-48.0, 48.0},
  {-48.0, 48.0},
  {-240.0, 240.0},
  {-240.0, 240.0},
  {-240.0, 240.0},
  {-96.0, 96.0},
  {-96.0, 96.0},
  {-96.0, 96.0},
  {-144.0, 144.0},
  {-144.0, 144.0},
  {-144.0, 144.0},
  {-24.0, 24.0},
  {-24.0, 24.0},
  {-24.0, 24.0},
  {-24.0, 24.0},
  {-24.0, 24.0},
  {-24.0, 24.0},
  {-240.0, 240.0},
  {-240.0, 240.0},
  {-240.0, 240.0},
  {-96.0, 96.0},
  {-96.0, 96.0},
  {-96.0, 96.0},
  {-144.0, 144.0},
  {-144.0, 144.0},
  {-144.0, 144.0},
  {-24.0, 24.0},
  {-24.0, 24.0},
  {-24.0, 24.0},
  {-24.0, 24.0},
  {-24.0, 24.0},
  {-24.0, 24.0}
};


std::vector<raisim::Vec<2>> JointLimits;
Eigen::VectorXd q;
Eigen::VectorXd q_dot;


  /// these variables are not in use. They are placed to show you how to create a random number sampler.
  thread_local static std::normal_distribution<double> normDist_;
  thread_local static std::uniform_real_distribution<double> uniDist_;
  thread_local static std::uniform_int_distribution<int> uniIntDist_;
  thread_local static std::mt19937 gen_;
};
thread_local std::mt19937 raisim::ENVIRONMENT::gen_;
thread_local std::uniform_real_distribution<double> raisim::ENVIRONMENT::uniDist_(0., 1.);
thread_local std::uniform_int_distribution<int> raisim::ENVIRONMENT::uniIntDist_(1, 5);
}

