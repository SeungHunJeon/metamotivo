//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#include "Environment.hpp"
#include "VectorizedEnvironment.hpp"

int THREAD_COUNT = 1;

using namespace raisim;

int main(int argc, char *argv[]) {
  RSFATAL_IF(argc != 3, "got "<<argc<<" arguments. "<<"This executable takes three arguments: 1. resource directory, 2. configuration file")

  std::string resourceDir(argv[1]), cfgFile(argv[2]);
  std::ifstream myfile (cfgFile);
  std::string config_str, line;
  bool escape = false;

  while (std::getline(myfile, line)) {
    if(line == "environment:") {
      escape = true;
      while (std::getline(myfile, line)) {
        if(line.substr(0, 2) == "  ")
          config_str += line.substr(2) + "\n";
        else if (line[0] == '#')
          continue;
        else
          break;
      }
    }
    if(escape)
      break;
  }
  config_str.pop_back();
  VectorizedEnvironment<ENVIRONMENT> vecEnv(resourceDir, config_str);

  Yaml::Node config;
  Yaml::Parse(config, config_str);

  EigenRowMajorMat observation(config["num_envs"].template As<int>(), vecEnv.getObDim());
  EigenRowMajorMat action(config["num_envs"].template As<int>(), vecEnv.getActionDim());
  EigenVec reward(config["num_envs"].template As<int>(), 1);
  EigenBoolVec dones(config["num_envs"].template As<int>(), 1);
  EigenRowMajorMat gc_init(config["num_envs"].template As<int>(), vecEnv.getActionDim()+7);
  EigenRowMajorMat gv_init(config["num_envs"].template As<int>(), vecEnv.getActionDim()+6);
  action.setZero();
  for (int i = 0 ; i < config["num_envs"].template As<int>(); i++) {
    gc_init.row(i).segment(3,4) << 1, 0, 0, 0;
  }

  Eigen::Ref<EigenRowMajorMat> ob_ref(observation), action_ref(action), gc_init_ref(gc_init), gv_init_ref(gv_init);
  Eigen::Ref<EigenVec> reward_ref(reward);
  Eigen::Ref<EigenBoolVec> dones_ref(dones);

  vecEnv.reset(gc_init_ref, gv_init_ref);
  vecEnv.step(action_ref, reward_ref, dones_ref);

  return 0;
}