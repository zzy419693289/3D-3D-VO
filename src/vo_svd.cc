#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include "vo_svd.h"
//������̻��õ�Opencv�Լ�eigen������
using namespace std;
//ʹ��SVD�ֽ���󣬿���ͨ��ƥ��õ�3D��ԣ����������֮֡���λ�˱仯R��t�������R��t�ǵ�һ֡���ڶ�֡�е������仯����ȻҲ���Կ���ڶ�֡����һ֡�ģ�λ�˱仯��
//����R=UV^T,t=p-Rp'������W=USV^T,W=Q'*Q^T,Q=P-p,Q'=P'-p��p�ǵ�һ֡���������꣬p'�ǵڶ�֡���������ꡣ
//���룺pts1ǰһ֡�����������������pts2����֢�����������������R�ڶ�֡����һ֡����̬��ת����3*3��t�ڶ�֡����һ֡��ƽ������3*1
void pose_svd_3d3d(const vector<cv::Point3f>& pts1, const vector<cv::Point3f>& pts2, cv::Mat& R, cv::Mat& t)
{
	cv::Point3f p1, p2;     // p1��p2����֡��������ά���ꡣ������ά����һ����mΪ��λ����������һ���õ��������㡣
	int N = pts1.size();	// ������֡ƥ��ĵ��������һ�µģ���������������ͬ
	//������������
	for (int i = 0; i<N; i++)	
	{
		p1 += pts1[i];	//ע������ķ��������ع��ģ�����ʵ����ά����Ķ�Ӧ�Ӽ�
		p2 += pts2[i];
	}
	p1 = cv::Point3f(p1 / N);
	p2 = cv::Point3f(p2 / N);
	//����ǰ����֡��q
	vector<cv::Point3f>     q1(N), q2(N); // ����vector��ʼ��
	for (int i = 0; i<N; i++)
	{
		q1[i] = pts1[i] - p1;
		q2[i] = pts2[i] - p2;
	}
	// ����W��W��һ��double���͵�3*3��������ʹ��Eigen����һ��3*3�ľ��󣬳�ʼֵΪ0����������ʵ����һ��������������ѣ��ⲻ�������ά�������о�������ӣ�
	Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
	for (int i = 0; i<N; i++)	//vector3dһ��ʼ���У�transpose�������Ϊ�У�������������������˻���һ��3*3�ľ��󡣣�Eigen�����vector��ʵ��N*1��mat�����Ժ�mat��ͬ���Ϳ�����ͬ�����ع����*���о�������㣩
	{
		W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();	//W��ǰ����֡������ά����Ļ�֮�͡����忴�Ƶ���ʽ
	}
	//cout << "W=" << W << endl;
	// ��W����SVD�ֽ⡣
	Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);	//Full��U��V��ʾ������ʵͨ������ÿһ����������U��V�����������Full����Thin�����ʾֻ�����м������ֵS�ԽǾ���
	Eigen::Matrix3d U = svd.matrixU();	//����������ֱ��ͨ��svdȥ����U��V�������S����
	Eigen::Matrix3d V = svd.matrixV();
	//cout << "U=" << U << endl;
	//cout << "V=" << V << endl;
	//ֱ�Ӽ���ڶ�֡����һ֡��R��t��R=UV^T,t=p-Rp'
	Eigen::Matrix3d R_ = U* (V.transpose());	//3*3��matrix
	Eigen::Vector3d t_ = Eigen::Vector3d(p2.x, p2.y, p2.z) - R_ * Eigen::Vector3d(p1.x, p1.y, p1.z);	//3*1��matrix

	// ����ת����cv::Mat��ע�⣬�����R��t�ǵڶ�֡����һ֡��λ�˱任������ʵ��R'��t'��ת����ϵ��R'=R^-1,t'=-R^-1*t��ͬʱ��ע������R������������R^-1=R^T��
	/*	
	R = (cv::Mat_<double>(3, 3) <<
		R_(0, 0), R_(0, 1), R_(0, 2),
		R_(1, 0), R_(1, 1), R_(1, 2),
		R_(2, 0), R_(2, 1), R_(2, 2)
		);
	t = (cv::Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
	*/
	R = (cv::Mat_<double>(3, 3) <<			//1 to 2, pose change
		R_(0, 0), R_(1, 0), R_(2, 0),
		R_(0, 1), R_(1, 1), R_(2, 1),
		R_(0, 2), R_(1, 2), R_(2, 2)
		);
	t = (cv::Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
	t =-1*R*t;
}
