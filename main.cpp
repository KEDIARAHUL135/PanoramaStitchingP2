#define _USE_MATH_DEFINES

#include "opencv2/opencv.hpp"
#include <iostream>
#include <filesystem>
#include <cmath>

namespace fs = std::filesystem;		// ISO C++17 Standard (/std:c++17)


int ReadImage(std::string ImageFolderPath, std::vector<cv::Mat>& Images)
{
	// Checking if path is of folder.
	if (fs::is_directory(fs::status(ImageFolderPath)))
	{
		std::vector<int> ImageNames_Int;
		std::vector<std::string> ImageNames;
		for (const auto& entry : fs::directory_iterator(ImageFolderPath))
		{
			ImageNames.push_back(entry.path().u8string());
			ImageNames_Int.push_back(stoi(entry.path().stem().u8string()));
		}

		// Sorting the images according to their names and reading in ascending order.
		std::vector<int> ImageNames_Int_Sorted = ImageNames_Int;
		std::sort(ImageNames_Int_Sorted.begin(), ImageNames_Int_Sorted.end());

		for (auto x : ImageNames_Int_Sorted)
		{
			for (int i = 0; i < ImageNames_Int.size(); i++)
			{
				if (ImageNames_Int[i] == x)
				{
					cv::Mat Image = cv::imread(ImageNames[i]);	// Reading images one by one.

					if (Image.empty())		// Checking if image is read
					{
						std::cout << "Not able to read image: " << ImageNames[i] << std::endl;
						exit(0);
					}

					Images.push_back(Image);

					break;
				}
			}
		}
	}

	else                                    // If it is not folder(Invalid Path).
		std::cout << "\nEnter valid Image Folder Path.\n";

	if (Images.size() < 2)
	{
		std::cout << "\nNot enough images found. Please provide 2 or more images.\n";
		exit(1); 
	}

	return 0;
}


void FindMatches(cv::Mat BaseImage, cv::Mat SecImage, std::vector<cv::DMatch>& GoodMatches, std::vector<cv::KeyPoint>& BaseImage_kp, std::vector<cv::KeyPoint>& SecImage_kp)
{
	using namespace cv;
	//using namespace cv::xfeatures2d;
	// Using SIFT to find the keypointsand decriptors in the images
	Ptr<SIFT> Sift = SIFT::create();
	cv::Mat BaseImage_des, SecImage_des;
	cv::Mat BaseImage_Gray, SecImage_Gray;
	cv::cvtColor(BaseImage, BaseImage_Gray, cv::COLOR_BGR2GRAY);
	cv::cvtColor(SecImage, SecImage_Gray, cv::COLOR_BGR2GRAY);
	Sift->detectAndCompute(BaseImage_Gray, cv::noArray(), BaseImage_kp, BaseImage_des);
	Sift->detectAndCompute(SecImage_Gray, cv::noArray(), SecImage_kp, SecImage_des);
	
	// Using Brute Force matcher to find matches.
	cv::BFMatcher BF_Matcher;
	std::vector<std::vector<cv::DMatch>> InitialMatches;
	BF_Matcher.knnMatch(BaseImage_des, SecImage_des, InitialMatches, 2);
	
	// Applying ratio test and filtering out the good matches.
	for (int i = 0; i < InitialMatches.size(); ++i)
	{
		if (InitialMatches[i][0].distance < 0.75 * InitialMatches[i][1].distance)
		{
			GoodMatches.push_back(InitialMatches[i][0]);
		}
	}
}


void FindHomography(std::vector<cv::DMatch> Matches, std::vector<cv::KeyPoint> BaseImage_kp, std::vector<cv::KeyPoint> SecImage_kp, cv::Mat& HomographyMatrix)
{
	// If less than 4 matches found, exit the code.
	if (Matches.size() < 4)
	{	
		std::cout << "\nNot enough matches found between the images.\n";
		exit(0);
	}
	// Storing coordinates of points corresponding to the matches found in both the images
	std::vector<cv::Point2f> BaseImage_pts, SecImage_pts;
	for (int i = 0 ; i < Matches.size() ; i++)
	{
		cv::DMatch Match = Matches[i];
		BaseImage_pts.push_back(BaseImage_kp[Match.queryIdx].pt);
		SecImage_pts.push_back(SecImage_kp[Match.trainIdx].pt);
	}

	// Finding the homography matrix(transformation matrix).
	HomographyMatrix = cv::findHomography(SecImage_pts, BaseImage_pts, cv::RANSAC, (4.0));
}



void GetNewFrameSizeAndMatrix(cv::Mat &HomographyMatrix, int* Sec_ImageShape, int* Base_ImageShape, int* NewFrameSize, int* Correction)
{
	// Reading the size of the image
	int Height = Sec_ImageShape[0], Width = Sec_ImageShape[1];

	// Taking the matrix of initial coordinates of the corners of the secondary image
	// Stored in the following format : [[x1, x2, x3, x4], [y1, y2, y3, y4], [1, 1, 1, 1]]
	// Where(xi, yi) is the coordinate of the i th corner of the image.
	double initialMatrix[3][4] = { {0, (double)Width - 1, (double)Width - 1, 0},
								   {0, 0, (double)Height - 1, (double)Height - 1},
								   {1.0, 1.0, 1.0, 1.0} };
	cv::Mat InitialMatrix = cv::Mat(3, 4, CV_64F, initialMatrix);// .inv();


	// Finding the final coordinates of the corners of the image after transformation.
	// NOTE: Here, the coordinates of the corners of the frame may go out of the
	// frame(negative values).We will correct this afterwards by updating the
	// homography matrix accordingly.
	cv::Mat FinalMatrix = HomographyMatrix * InitialMatrix;

	cv::Mat x = FinalMatrix(cv::Rect(0, 0, FinalMatrix.cols, 1));
	cv::Mat y = FinalMatrix(cv::Rect(0, 1, FinalMatrix.cols, 1));
	cv::Mat c = FinalMatrix(cv::Rect(0, 2, FinalMatrix.cols, 1));

	
	cv::Mat x_by_c = x.mul(1 / c);
	cv::Mat y_by_c = y.mul(1 / c);

	// Finding the dimentions of the stitched image frame and the "Correction" factor
	double min_x, max_x, min_y, max_y;
	cv::minMaxLoc(x_by_c, &min_x, &max_x);
	cv::minMaxLoc(y_by_c, &min_y, &max_y);
	min_x = (int)round(min_x); max_x = (int)round(max_x);
	min_y = (int)round(min_y); max_y = (int)round(max_y);

	
	int New_Width = max_x, New_Height = max_y;
	Correction[0] = 0; Correction[1] = 0;
	if (min_x < 0)
	{
		New_Width -= min_x;
		Correction[0] = abs(min_x);
	}
	if (min_y < 0)
	{
		New_Height -= min_y;
		Correction[1] = abs(min_y);
	}

	// Again correcting New_Widthand New_Height
	// Helpful when secondary image is overlaped on the left hand side of the Base image.
	New_Width = (New_Width < Base_ImageShape[1] + Correction[0]) ? Base_ImageShape[1] + Correction[0] : New_Width;
	New_Height = (New_Height < Base_ImageShape[0] + Correction[1]) ? Base_ImageShape[0] + Correction[1] : New_Height;


	// Finding the coordinates of the corners of the image if they all were within the frame.
	cv::add(x_by_c, Correction[0], x_by_c);
	cv::add(y_by_c, Correction[1], y_by_c);


	cv::Point2f OldInitialPoints[4], NewFinalPonts[4];
	OldInitialPoints[0] = cv::Point2f(0, 0);
	OldInitialPoints[1] = cv::Point2f(Width - 1, 0);
	OldInitialPoints[2] = cv::Point2f(Width - 1, Height - 1);
	OldInitialPoints[3] = cv::Point2f(0, Height - 1);
	for (int i = 0; i < 4; i++) 
		NewFinalPonts[i] = cv::Point2f(x_by_c.at<double>(0, i), y_by_c.at<double>(0, i));


	// Updating the homography matrix.Done so that now the secondary image completely
	// lies inside the frame
	HomographyMatrix = cv::getPerspectiveTransform(OldInitialPoints, NewFinalPonts);

	// Setting variable for returning
	NewFrameSize[0] = New_Height; NewFrameSize[1] = New_Width;

}


void Convert_xy(std::vector<int> ti_x, std::vector<int> ti_y, std::vector<float>& xt, std::vector<float>& yt, int center_x, int center_y, int f)
{
	for (int i = 0; i < ti_y.size(); i++)
	{
		xt.push_back((f * tan((float)(ti_x[i] - center_x) / f)) + center_x);
		yt.push_back(((float)(ti_y[i] - center_y) / cos((float)(ti_x[i] - center_x) / f)) + center_y);
	}
}

// Used this website for code
// https://docs.opencv.org/master/d7/dff/tutorial_how_to_use_OpenCV_parallel_for_.html
class ParallelMandelbrot : public cv::ParallelLoopBody
{
public:
	ParallelMandelbrot(cv::Mat& TransformedImage, cv::Mat& InitialImage, std::vector<int> ti_x, std::vector<int> ti_y, std::vector<int> ii_tl_x, std::vector<int> ii_tl_y, std::vector<float> weight_tl, std::vector<float> weight_tr, std::vector<float> weight_bl, std::vector<float> weight_br)
		: TransformedImage(TransformedImage), InitialImage(InitialImage), ti_x(ti_x), ti_y(ti_y), ii_tl_x(ii_tl_x), ii_tl_y(ii_tl_y), weight_tl(weight_tl), weight_tr(weight_tr), weight_bl(weight_bl), weight_br(weight_br)
	{
	}
	virtual void operator ()(const cv::Range& range) const CV_OVERRIDE
	{
		for (int i = range.start; i < range.end; i++)
		{
			// https://stackoverflow.com/questions/7899108/opencv-get-pixel-channel-value-from-mat-image

			cv::Vec3b& TransformedImage_intensity = TransformedImage.at<cv::Vec3b>(ti_y[i], ti_x[i]);
			cv::Vec3b& InitialImage_intensity_tl = InitialImage.at<cv::Vec3b>(ii_tl_y[i], ii_tl_x[i]);
			cv::Vec3b& InitialImage_intensity_tr = InitialImage.at<cv::Vec3b>(ii_tl_y[i], ii_tl_x[i] + 1);
			cv::Vec3b& InitialImage_intensity_bl = InitialImage.at<cv::Vec3b>(ii_tl_y[i] + 1, ii_tl_x[i]);
			cv::Vec3b& InitialImage_intensity_br = InitialImage.at<cv::Vec3b>(ii_tl_y[i] + 1, ii_tl_x[i] + 1);

			for (int k = 0; k < InitialImage.channels(); k++)
			{
				TransformedImage_intensity.val[k] = ( weight_tl[i] * InitialImage_intensity_tl.val[k] ) +
													( weight_tr[i] * InitialImage_intensity_tr.val[k] ) +
													( weight_bl[i] * InitialImage_intensity_bl.val[k] ) +
													( weight_br[i] * InitialImage_intensity_br.val[k] );
			}
		}
	}
	ParallelMandelbrot& operator=(const ParallelMandelbrot&) {
		return *this;
	};
private:
	cv::Mat& TransformedImage;
	cv::Mat& InitialImage;
	std::vector<int> ti_x;
	std::vector<int> ti_y;
	std::vector<int> ii_tl_x;
	std::vector<int> ii_tl_y;
	std::vector<float> weight_tl;
	std::vector<float> weight_tr;
	std::vector<float> weight_bl;
	std::vector<float> weight_br;
};


void ProjectOntoCylinder(cv::Mat InitialImage, cv::Mat& TransformedImage, std::vector<int>& mask_x, std::vector<int>& mask_y)
{
	int h = InitialImage.rows, w = InitialImage.cols;
	int center_x = w / 2, center_y = h / 2;
	int f = 1100;			// 1100 field; 1000 Sun; 1500 Rainier; 1050 Helens

	// Creating a blank transformed image.
	TransformedImage = cv::Mat::zeros(cv::Size(InitialImage.cols, InitialImage.rows), InitialImage.type());

	// Storing all coordinates of the transformed image in 2 arrays (x and y coordinates)
	std::vector<int> ti_x, ti_y;
	for (int j = 0; j < h; j++)
	{
		for (int i = 0; i < w; i++)
		{
			ti_x.push_back(i);
			ti_y.push_back(j);
		}
	}

	// Finding corresponding coordinates of the transformed image in the initial image
	std::vector<float> ii_x, ii_y;
	Convert_xy(ti_x, ti_y, ii_x, ii_y, center_x, center_y, f);

	// Rounding off the coordinate values to get exact pixel values (top-left corner).
	std::vector<int> ii_tl_x, ii_tl_y;
	for (int i = 0; i < ii_x.size(); i++)
	{
		ii_tl_x.push_back((int)ii_x[i]);
		ii_tl_y.push_back((int)ii_y[i]);
	}

	// Finding transformed image points whose corresponding
	// initial image points lies inside the initial image
	std::vector<bool> GoodIndices;
	for (int i = 0; i < ii_tl_x.size(); i++)
		GoodIndices.push_back((ii_tl_x[i] >= 0) && (ii_tl_x[i] <= (w - 2)) && (ii_tl_y[i] >= 0) && (ii_tl_y[i] <= (h - 2)));
	
	// Removing all the outside points from everywhere
	ti_x.erase(std::remove_if(ti_x.begin(), ti_x.end(), [&GoodIndices, &ti_x](auto const& i) { return !GoodIndices.at(&i - ti_x.data()); }), ti_x.end());
	ti_y.erase(std::remove_if(ti_y.begin(), ti_y.end(), [&GoodIndices, &ti_y](auto const& i) { return !GoodIndices.at(&i - ti_y.data()); }), ti_y.end());
	
	ii_x.erase(std::remove_if(ii_x.begin(), ii_x.end(), [&GoodIndices, &ii_x](auto const& i) { return !GoodIndices.at(&i - ii_x.data()); }), ii_x.end());
	ii_y.erase(std::remove_if(ii_y.begin(), ii_y.end(), [&GoodIndices, &ii_y](auto const& i) { return !GoodIndices.at(&i - ii_y.data()); }), ii_y.end());
	
	ii_tl_x.erase(std::remove_if(ii_tl_x.begin(), ii_tl_x.end(), [&GoodIndices, &ii_tl_x](auto const& i) { return !GoodIndices.at(&i - ii_tl_x.data()); }), ii_tl_x.end());
	ii_tl_y.erase(std::remove_if(ii_tl_y.begin(), ii_tl_y.end(), [&GoodIndices, &ii_tl_y](auto const& i) { return !GoodIndices.at(&i - ii_tl_y.data()); }), ii_tl_y.end());

	// Bilinear interpolation
	std::vector<float> dx(ii_x.size()), dy(ii_y.size());
	std::transform(ii_x.begin(), ii_x.end(), ii_tl_x.begin(), dx.begin(), std::minus<float>());
	std::transform(ii_y.begin(), ii_y.end(), ii_tl_y.begin(), dy.begin(), std::minus<float>());

	std::vector<float> weight_tl, weight_tr, weight_bl, weight_br;
	for (int i = 0; i < dx.size(); i++)
	{
		weight_tl.push_back((1.0 - dx[i]) * (1.0 - dy[i]));
		weight_tr.push_back((dx[i]) * (1.0 - dy[i]));
		weight_bl.push_back((1.0 - dx[i]) * (dy[i]));
		weight_br.push_back((dx[i]) * (dy[i]));
	}

	// Used this website for code
	// https://docs.opencv.org/master/d7/dff/tutorial_how_to_use_OpenCV_parallel_for_.html
	ParallelMandelbrot parallelMandelbrot(TransformedImage, InitialImage, ti_x, ti_y, ii_tl_x, ii_tl_y, weight_tl, weight_tr, weight_bl, weight_br);
	cv::parallel_for_(cv::Range(0, weight_tl.size()), parallelMandelbrot);

	// Getting x coorinate to remove black region from rightand left in the transformed image
	int min_x = *min_element(ti_x.begin(), ti_x.end());

	// Cropping out the black region from both sides(using symmetricity)
	TransformedImage(cv::Rect(min_x, 0, TransformedImage.cols - min_x*2, TransformedImage.rows)).copyTo(TransformedImage);

	// Setting return values
	// mask_x = ti_x - min_x
	std::vector<int> min_x_v(ti_x.size(), min_x);
	std::transform(ti_x.begin(), ti_x.end(), min_x_v.begin(), std::back_inserter(mask_x),
		[](int a, int b) { return (a - b); });
	//mask_y = ti_y
	mask_y = ti_y;
}


cv::Mat StitchImages(cv::Mat BaseImage, cv::Mat SecImage)
{
	// Applying Cylindrical projection on SecImage
	cv::Mat SecImage_Cyl;
	std::vector<int> mask_x, mask_y;
	ProjectOntoCylinder(SecImage, SecImage_Cyl, mask_x, mask_y);

	// Getting SecImage Mask
	cv::Mat SecImage_Mask = cv::Mat::zeros(SecImage_Cyl.rows, SecImage_Cyl.cols, SecImage_Cyl.type());
	//std::vector<int> white_color(255, SecImage_Mask.channels());
	for (int i = 0; i < mask_x.size(); i++)
	{
		cv::Vec3b& value = SecImage_Mask.at<cv::Vec3b>(mask_y[i], mask_x[i]);
		for (int k = 0; k < SecImage_Cyl.channels(); k++)
			value.val[k] = 255;
	}

	// Finding matches between the 2 images and their keypoints
	std::vector<cv::DMatch> Matches;
	std::vector<cv::KeyPoint> BaseImage_kp, SecImage_kp;
	FindMatches(BaseImage, SecImage_Cyl, Matches, BaseImage_kp, SecImage_kp);

	// Finding homography matrix.
	cv::Mat HomographyMatrix;
	FindHomography(Matches, BaseImage_kp, SecImage_kp, HomographyMatrix);

	// Finding size of new frame of stitched images and updating the homography matrix
	int Sec_ImageShape[2] = { SecImage_Cyl.rows, SecImage_Cyl.cols };
	int Base_ImageShape[2] = { BaseImage.rows, BaseImage.cols };
	int NewFrameSize[2], Correction[2];
	GetNewFrameSizeAndMatrix(HomographyMatrix, Sec_ImageShape, Base_ImageShape, NewFrameSize, Correction);

	// Finally placing the images upon one another.
	cv::Mat SecImage_Transformed, SecImage_Transformed_Mask;
	cv::warpPerspective(SecImage_Cyl, SecImage_Transformed, HomographyMatrix, cv::Size(NewFrameSize[1], NewFrameSize[0]));
	cv::warpPerspective(SecImage_Mask, SecImage_Transformed_Mask, HomographyMatrix, cv::Size(NewFrameSize[1], NewFrameSize[0]));

	cv::Mat BaseImage_Transformed = cv::Mat::zeros(NewFrameSize[0], NewFrameSize[1], BaseImage.type());
	BaseImage.copyTo(BaseImage_Transformed(cv::Rect(Correction[0], Correction[1], BaseImage.cols, BaseImage.rows)));

	cv::Mat StitchedImage, temp;
	cv::bitwise_not(SecImage_Transformed_Mask, SecImage_Transformed_Mask);
	cv::bitwise_and(BaseImage_Transformed, SecImage_Transformed_Mask, temp);
	cv::bitwise_or(SecImage_Transformed, temp, StitchedImage);

	return StitchedImage;
}


int main()
{
	// Reading images.
	std::vector<cv::Mat> Images;		// Input Images will be stored in this list.
	ReadImage("InputImages/Field", Images);

	cv::Mat BaseImage, TransformedImage;
	std::vector<int> dummy1, dummy2;
	ProjectOntoCylinder(Images[0], BaseImage, dummy1, dummy2);

	for (int i = 1; i < Images.size(); i++)
	{
		cv::Mat StitchedImage = StitchImages(BaseImage, Images[i]);

		StitchedImage.copyTo(BaseImage);
	}
	
	cv::imwrite("cpp_Stitched_Panorama.png", BaseImage);

	return 0;
}