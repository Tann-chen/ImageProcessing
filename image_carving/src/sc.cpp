#include "sc.h"

using namespace cv;
using namespace std;


bool seam_carving(Mat& in_image, int new_width, int new_height, Mat& out_image){

    // some sanity checks
    // Check 1 -> new_width <= in_image.cols
    if(new_width>in_image.cols){
        cout<<"Invalid request!!! new_width has to be smaller than the current size!"<<endl;
        return false;
    }

    if(new_height>in_image.rows){
        cout<<"Invalid request!!! ne_height has to be smaller than the current size!"<<endl;
        return false;
    }
    
    if(new_width<=0){
        cout<<"Invalid request!!! new_width has to be positive!"<<endl;
        return false;
    }
    
    if(new_height<=0){
        cout<<"Invalid request!!! new_height has to be positive!"<<endl;
        return false;
    }

    return seam_carving_trivial(in_image, new_width, new_height, out_image);
}



bool seam_carving_trivial(Mat& in_image, int new_width, int new_height, Mat& out_image){
    Mat iimage = in_image.clone();
    Mat oimage = in_image.clone();

    while(iimage.rows!=new_height || iimage.cols!=new_width){
        // horizontal seam if needed
        if(iimage.rows>new_height){
            reduce_horizontal_seam(iimage, oimage);
            iimage = oimage.clone();
        }
        // vertical seam if needed
        if(iimage.cols>new_width){
            reduce_vertical_seam(iimage, oimage);
            iimage = oimage.clone();
        }
    }
    
    out_image = oimage.clone();
    return true;
}


void compute_energy_by_gradient(Mat& in_image, Mat& magn_img){
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    Mat img_gray;
    Mat iin_image = in_image.clone();

    GaussianBlur(iin_image, iin_image, Size(3, 3), 0, 0, BORDER_DEFAULT);

    // Convert it to gray
    cvtColor(iin_image, img_gray, CV_BGR2GRAY);

    // Generate grad_x and grad_y
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    // Gradient X
    Scharr(img_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT);
    //Sobel(img_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);

    // Gradient Y
    Scharr(img_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT);
    //Sobel(img_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(grad_y, abs_grad_y);

    // Total Gradient (approximate)
    Mat imagn_img;
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, imagn_img);
    imshow("temp_img", imagn_img);
    waitKey(300);
    magn_img = imagn_img.clone();
}


vector<int> find_horizontal_seam_by_dp(Mat &in_image) {
    Mat magn_img;
    compute_energy_by_gradient(in_image, magn_img);

    int rows = magn_img.rows;
    int cols = magn_img.cols;

    int T[rows][cols];
    char Dir[rows][cols];


    //first col
    for (int r = 0; r < rows; ++r) {
        T[r][0] = 0;
        Dir[r][0] = '*';
    }

    //after first col
    for (int c = 1; c < cols; ++c) {
        for (int r = 0; r < rows; ++r) {
            uchar self = magn_img.at<uchar>(r, c);
            //left
            int temp = abs(magn_img.at<uchar>(r, c - 1) - self) + T[r][c - 1];
            char tempChar = '<';

            //left up
            if (r > 0 && abs(magn_img.at<uchar>(r - 1, c - 1) - self) + T[r - 1][c - 1] < temp) {
                temp = abs(magn_img.at<uchar>(r - 1, c - 1) - self) + T[r - 1][c - 1];
                tempChar = '^';
            }
            //left down
            if (r < rows - 1 && abs(magn_img.at<uchar>(r + 1, c - 1) - self) + T[r + 1][c - 1] < temp) {
                temp = abs(magn_img.at<uchar>(r + 1, c - 1) - self) + T[r + 1][c - 1];
                tempChar = 'V';
            }

            T[r][c] = temp;
            Dir[r][c] = tempChar;
        }
    }

    /*
    //first col
    for (int r = 0; r < rows; ++r) {
        T[r][0] = magn_img.at<uchar>(r,0);
        Dir[r][0] = '*';
    }

    //after first col
    for (int c = 1; c < cols; ++c) {
        for (int r = 0; r < rows; ++r) {
            uchar self = magn_img.at<uchar>(r, c);
            //left
            int temp = magn_img.at<uchar>(r, c - 1) + self;
            char tempChar = '<';

            //left up
            if (r != 0 && magn_img.at<uchar>(r - 1, c - 1) + self < temp) {
                temp = magn_img.at<uchar>(r - 1, c - 1) + self;
                tempChar = '^';
            }
            //left down
            if (r != rows - 1 && magn_img.at<uchar>(r + 1, c - 1) + self < temp) {
                temp = magn_img.at<uchar>(r + 1, c - 1) + self;
                tempChar = 'V';
            }

            T[r][c] = temp;
            Dir[r][c] = tempChar;
        }
    }
    */


    //find the solution
    int index = 0;
    int temp = T[0][cols - 1];
    for (int r = 1; r < rows; ++r) {
        if (T[r][cols - 1] < temp) {
            temp = T[r][cols - 1];
            index = r;
        }
    }

    vector<int> indexes;
    int tempCol = cols - 1;
    while (tempCol >= 0) {
        indexes.push_back(index);
        char tempFlag = Dir[index][tempCol];
        if (tempFlag == '^') {
            index --;
        } else if (tempFlag == 'V') {
            index ++;
        }
        tempCol--;
    }


    //reverse the vector
    vector<int> removedIndexes;
    for (long i = indexes.size() - 1; i >= 0; --i) {
        int temp = indexes[i];
        removedIndexes.push_back(temp);
    }

    return removedIndexes;
}


vector<int> find_vertical_seam_by_dp(Mat &in_image) {
    Mat magn_img;     //float matrix
    compute_energy_by_gradient(in_image, magn_img);

    int rows = magn_img.rows;
    int cols = magn_img.cols;

    int T[rows][cols];
    char Dir[rows][cols];


    //first line
    for (int c = 0; c < cols; ++c) {
        T[0][c] = 0;
        Dir[0][c] = '*';
    }

    //after first line
    for (int r = 1; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            uchar self = magn_img.at<uchar>(r, c);

            int temp = abs(magn_img.at<uchar>(r - 1, c) - self) + T[r - 1][c];
            char tempDir = '^';

            //up left
            if (c > 0 && abs(magn_img.at<uchar>(r - 1, c - 1) - self) + T[r - 1][c - 1] < temp) {
                temp = abs(magn_img.at<uchar>(r - 1, c - 1) - self) + T[r - 1][c - 1];
                tempDir = '<';
            }

            //up right
            if (c != cols - 1 &&
                abs(magn_img.at<uchar>(r - 1, c + 1) - self) + T[r - 1][c + 1] < temp) {
                temp = abs(magn_img.at<uchar>(r - 1, c + 1) - self) + T[r - 1][c + 1];
                tempDir = '>';
            }

            //record
            T[r][c] = temp;
            Dir[r][c] = tempDir;
        }
    }

    /*
    //first line
    for (int c = 0; c < cols; ++c) {
        T[0][c] = magn_img.at<uchar>(0,c);
        Dir[0][c] = '*';
    }

    //after first line
    for (int r = 1; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            uchar self = magn_img.at<uchar>(r, c);
            int temp = self + magn_img.at<uchar>(r - 1, c);
            char tempDir = '^';

            //up left
            if (c != 0 && self + magn_img.at<uchar>(r - 1, c - 1) < temp) {
                temp = self + magn_img.at<uchar>(r - 1, c - 1);
                tempDir = '<';
            }

            //up right
            if (c != cols - 1 && self + magn_img.at<uchar>(r - 1, c + 1) < temp) {
                temp = self + magn_img.at<uchar>(r - 1, c + 1);
                tempDir = '>';
            }

            //record
            T[r][c] = temp;
            Dir[r][c] = tempDir;
        }
    }
    */

    //find the solution
    int index = 0;
    int temp = T[rows-1][0];
    for (int c = 1; c < cols; ++c) {
        if (T[rows - 1][c] < temp) {
            temp = T[rows - 1][c];
            index = c;
        }
    }

    vector<int> indexes;
    int tempRow = rows - 1;
    while (tempRow >= 0) {
        indexes.push_back(index);
        char tempFlag = Dir[tempRow][index];
        if (tempFlag == '<') {
            index --;
        } else if (tempFlag == '>') {
            index ++;
        }
        tempRow--;
    }


    //reverse the vector
    vector<int> removedIndexes;
    for (long i = indexes.size() - 1; i >= 0; --i) {
        int temp = indexes[i];
        removedIndexes.push_back(temp);
    }

    return removedIndexes;
}

void reduce_horizontal_seam(Mat &in_image, Mat &out_image) {
    int rows = in_image.rows - 1;
    int cols = in_image.cols;

    //find seam
    vector<int> removedIndexes = find_horizontal_seam_by_dp(in_image);


    Mat oout_image = Mat(rows, cols, CV_8UC3);

    for (int c = 0; c < cols; ++c) {
        int removedTarget = removedIndexes[c];

        for (int r = 0; r < removedTarget; ++r) {
            Vec3b pixel = in_image.at<Vec3b>(r, c);
            oout_image.at<Vec3b>(r, c) = pixel;
        }

        for (int r = removedTarget + 1; r < rows + 1; ++r) {
            Vec3b pixel = in_image.at<Vec3b>(r, c);
            oout_image.at<Vec3b>(r - 1, c) = pixel;
        }
    }

    out_image = oout_image.clone();
    //cout << "cut one horizontal_seam" <<endl;
}


void reduce_vertical_seam(Mat &in_image, Mat &out_image) {
    int rows = in_image.rows;
    int cols = in_image.cols - 1;

    vector<int> removedIndexes = find_vertical_seam_by_dp(in_image);

    Mat oout_image = Mat(rows, cols, CV_8UC3);

    for (int r = 0; r < rows; ++r){
        int removedTarget = removedIndexes[r];

        for (int c = 0; c < removedTarget; ++c) {
            Vec3b pixel = in_image.at<Vec3b>(r, c);
            oout_image.at<Vec3b>(r, c) = pixel;
        }

        for (int c = removedTarget+1; c < cols+1; ++c) {
            Vec3b pixel = in_image.at<Vec3b>(r, c);
            oout_image.at<Vec3b>(r, c-1) = pixel;
        }
    }

    out_image = oout_image.clone();
    //cout << "cut one vertical seam" <<endl;
}

