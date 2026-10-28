/*
RGB-D VIO for UAV Pest Control — UCLA
Entry point: mirrors main.cpp of FAST-LIVO2.
*/

#include "RGBDMapper.h"

int main(int argc, char **argv)
{
  ros::init(argc, argv, "rgbdMapping");
  ros::NodeHandle nh;           // global: used for pub/sub topic names
  ros::NodeHandle pnh("~");     // private: resolves params under /rgbdMapping/
  image_transport::ImageTransport it(nh);

  RGBDMapper mapper(pnh);       // readParameters + loadVoxelConfig use private ns
  mapper.initializeSubscribersAndPublishers(nh, it);
  mapper.run();

  return 0;
}
