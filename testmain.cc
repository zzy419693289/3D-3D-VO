#include "CPU_ORB.h"
#include "GPU_ORB.h"
#include "vo_svd.h"
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>
using namespace std;
int main(int argc, char **argv)
{
	//check open right
	if (argc != 2)
	{
		cerr << endl << "Usage: ./XXX path_to_settings.yaml" << endl;
		return 1;
	}
	/////////////////////////////////////////////////////////// ZED�����ʼ��//////////////////////////////////////////////////////////////////////
	//Check settings file
	const string &strSettingsFile = argv[1];	//��Ϊ./���ҲҪ��һ��argv���������ǵ����붼�Ǵ�1��ʼ��
	cv::FileStorage fsRead(strSettingsFile.c_str(), cv::FileStorage::READ);	//��һ��������ʾ���ļ������ַ�������ʽ���������string���ڶ���������ָ�����ļ��Ƕ�����д��
	if (!fsRead.isOpened())
	{
		cerr << "Failed to open settings file at: " << strSettingsFile << endl;
		exit(-1);
	}
	//������
	double Camerafx, Camerafy, Cameracx, Cameracy;
	fsRead["Camera.fx"] >> Camerafx;
	fsRead["Camera.fy"] >> Camerafy;
	fsRead["Camera.cx"] >> Cameracx;
	fsRead["Camera.cy"] >> Cameracy;

	fsRead.release();	//��ȡ�������ͷŲ��ر��ļ���

	sl::Camera zed;
	// Set configuration parameters
	sl::InitParameters init_params;
	init_params.depth_mode = sl::DEPTH_MODE_PERFORMANCE;
	init_params.camera_resolution = sl::RESOLUTION_HD720; // Use 1280*720  mode
	init_params.coordinate_units = sl::UNIT_METER;	//unit is m
	init_params.camera_fps = 30; // SLAM is 15 fps
	// Open the camera
	sl::ERROR_CODE err = zed.open(init_params);
	if (err != sl::SUCCESS)
	{
		zed.close();
		exit(-1);
	}
	sl::Mat zed_leftimage(zed.getResolution(), sl::MAT_TYPE_8U_C4);
	sl::Mat zed_depthimage(zed.getResolution(), sl::MAT_TYPE_32F_C1);
	//cv_Mat use zed_Mat's memory
	cv::Mat imleftRGB = cv::Mat((int)zed_leftimage.getHeight(), (int)zed_leftimage.getWidth(), CV_8UC4, zed_leftimage.getPtr<sl::uchar1>(sl::MEM_CPU));
	cv::Mat imD = cv::Mat((int)zed_depthimage.getHeight(), (int)zed_depthimage.getWidth(), CV_32FC1, zed_depthimage.getPtr<sl::float1>(sl::MEM_CPU));
	cv::Mat imleftRGB1, imleftRGB2; 
	cv::Mat imD1, imD2; 

	for (int i = 0; i<10; i++)	//Eliminate the overexposure of the camera. It will happen at camera firstly open.
	{
		if (zed.grab() == sl::SUCCESS)
		{
			// A new image is available if grab() returns SUCCESS
			zed.retrieveImage(zed_leftimage, sl::VIEW_LEFT, sl::MEM_CPU); // Get the left image, TX2 have no GPU memory
			zed.retrieveMeasure(zed_depthimage, sl::MEASURE_DEPTH, sl::MEM_CPU); 	//get depth image, use GPU
		}
		usleep(5000);	//wait 5ms
	}
	
	///////////////////////////////////////////////////////////��������ͼ������ͨ�����س�������ͼ��ѡ��///////////////////////////////////////////////////////////////
	while (cv::waitKey(1) != '\r')
	{
		if (zed.grab() == sl::SUCCESS)
		{
			// A new image is available if grab() returns SUCCESS
			zed.retrieveImage(zed_leftimage, sl::VIEW_LEFT, sl::MEM_CPU); // Get the left image, TX2 have no GPU memory
			zed.retrieveMeasure(zed_depthimage, sl::MEASURE_DEPTH, sl::MEM_CPU); 	//get depth image, use GPU
		}
		cv::imshow("image",imleftRGB);
	}
	imleftRGB1 = imleftRGB.clone();	//���
	imD1 = imD.clone();
	cout << "Fisrt Capture!" << endl;

	while (cv::waitKey(1) != '\r')
	{
		if (zed.grab() == sl::SUCCESS)
		{
			// A new image is available if grab() returns SUCCESS
			zed.retrieveImage(zed_leftimage, sl::VIEW_LEFT, sl::MEM_CPU); // Get the left image, TX2 have no GPU memory
			zed.retrieveMeasure(zed_depthimage, sl::MEASURE_DEPTH, sl::MEM_CPU); 	//get depth image, use GPU
		}
		cv::imshow("image", imleftRGB);
	}
	imleftRGB2 = imleftRGB.clone();	//���
	imD2 = imD.clone();
	cout << "Second Capture!" << endl;

	///////////////////////////////////////////////////////////��������ͼ���ORB������ÿ��ͼ����ȡ���1000��������//////////////////////////////////////////////////////
	ORB_CPU orb1(imleftRGB1, 1000);
	ORB_CPU orb2(imleftRGB2, 1000);
	//�ֱ����ƥ��
	vector<cv::DMatch> matches = ORB_CPU::MatchPic(orb1, orb2, false, true);	//�˲�ƥ��,ͬʱ��ƥ�������м�¼
	//DMatch����ṹ���¼��һ��ƥ��õ�orb������������������������queryIdx��	trainIdx�ֱ��Ӧ��ǰ������ͼ��keypoints��vector<cv::KeyPoint>����š��������ǿ��Խ�ƥ��õĵ�ÿһ���������������������������
	//������֡����ά���Ƽ�
	vector<cv::Point3f> pointcloud1;	
	vector<cv::Point3f> pointcloud2;
	for (int i = 0; i < matches.size(); i++)	//����matches��ͬʱ����Ȳ����ĵ�����ų���������ƥ��
	{
		//�γɵ�һ֡�и�ƥ������ά���ꡣע���ά����ά����Ļָ�����Ҫ�õ�����ڲ�
		cv::KeyPoint KP1 = orb1.keypoints[matches[i].queryIdx];	//queryIdxǰһ֡���������š�������ȡ��i��ƥ����У�ͼһ��������
		cv::Point2f point1= KP1.pt;	//��ȡ�õ��ͼ�����ꡣ������x��y��ע��x��Ӧ��ͼ�������n��y��Ӧ��m��
		float z1 = imD1.at<float>((int)point1.y, (int)point1.x);
		float x1 = z1*(point1.x - Cameracx) / Camerafx;
		float y1 = z1*(point1.y - Cameracy) / Camerafy;
		if (z1<0.5 || z1>5 || std::isnan(z1))	//check z is not nan. nan means this point have no data
			continue;
		//�γɵڶ�֡�и�ƥ������ά���ꡣ
		cv::KeyPoint KP2 = orb2.keypoints[matches[i].trainIdx];	//trainIdx��һ֡����������
		cv::Point2f point2 = KP2.pt;	//��ȡ�õ��ͼ�����ꡣ������x��y��ע��x��Ӧ��ͼ�������n��y��Ӧ��m��
		float z2 = imD2.at<float>((int)point2.y, (int)point2.x);
		float x2 = z2*(point2.x - Cameracx) / Camerafx;
		float y2 = z2*(point2.y - Cameracy) / Camerafy;
		if (z2<0.5 || z2>5 || std::isnan(z2))
			continue;

		cv::Point3f tmp1(x1, y1, z1);
		pointcloud1.push_back(tmp1);
		cv::Point3f tmp2(x2, y2, z2);
		pointcloud2.push_back(tmp2);

		//cout<<"x1 y1 z1"<<endl<<x1<<" "<<y1<<" "<<z1<<endl;
		//cout<<"x2 y2 z2"<<endl<<x2<<" "<<y2<<" "<<z2<<endl;
	}
	
	///////////////////////////////////////////////////////////////////////////////////����ڶ�֡����һ֡��λ�˱任//////////////////////////////////////////////////////////////////////////////////////
	cv::Mat R, t;	//����ڶ�֡����һ֡����ת��ƽ�ƾ���
	cv::Vec3d Eal;
	pose_svd_3d3d(pointcloud1, pointcloud2, R, t, Eal);
	cout<<"1 to 2 R is:"<<endl;
	cout<< R<<endl;
	cout<<"1 to 2 R-Y-P is:"<<endl;
	cout<< Eal <<endl;
	cout<<"t is:"<<endl;
	cout<< t<<endl;

	///////////////////////////////////////////////////////////////////////////////// Close the camera///////////////////////////////////////////////////////////////////////
	while (cv::waitKey(1) != '\r');	//���س��˳����г���
	cv::destroyAllWindows();	//������ͼ�񴰿ڹر�
	zed_leftimage.free(sl::MEM_CPU);
	zed_depthimage.free(sl::MEM_CPU);
	zed.close();

	return 0;
}



