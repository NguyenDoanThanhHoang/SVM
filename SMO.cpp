#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#define CLEAR_TEXT "\33[0m"
#define NORMAL_TEXT "\33[1;37m"
#define RED_TEXT "\33[1;31m"
#define GREEN_TEXT "\33[1;32m"
#define YELLOW_TEXT "\33[1;33m"
#define BLUE_TEXT "\33[1;34m"
#define PINK_TEXT "\33[1;35m"
#define CYAN_TEXT "\33[1;36m"
#define WHITE_TEXT "\33[1;37m"
#define GREY_TEXT "\033[38;5;243m"
using namespace std;
const double tol = 1e-3;
const double eps = 1e-5;
const double C = 1.0; // Thông số điều chỉnh
struct Record
{
    float sepal_width, sepal_length, petal_width, petal_length;
    std::string classname;
    void reset_Record()
    {
        sepal_width = 0;
        sepal_length = 0;
        petal_width = 0;
        petal_length = 0;
        classname.clear();
    }
    void cout_Record()
    {
        std::cout << sepal_length << " " << sepal_width << " " << petal_length << " " << petal_width << " " << classname
                  << "\n";
    }
};
void createRecordVector(std::vector<Record> &records, const char *filename)
{
    // open file
    std::ifstream f(filename);
    // error checking
    if (!f)
    {
        std::cout << "Error\n" << endl;
        return;
    }
    cout << " DA DOC FILE THANH CONG!!!\n";
    // cout << "\nDU LIEU DUOC NHAP TU FILE LA: \n";
    std::string buffer;
    getline(f, buffer);
    getline(f, buffer);
    getline(f, buffer);
    Record buffer_record;
    records.clear();
    buffer_record.reset_Record();
    char buffer_char;
    // int STT = 0;
    while (true)
    {
        f >> buffer_record.sepal_length >> buffer_record.sepal_width >> buffer_record.petal_length >>
            buffer_record.petal_width >> buffer_record.classname;
        if (!buffer_record.sepal_length)
            break;
        records.push_back(buffer_record);
        // cout << ++STT << ": ";
        // records[records.size() - 1].cout_Record();
        buffer_record.reset_Record();
    }
    // close file
    f.close();
}
struct SVM
{
    vector<double> alpha;
    vector<int> tickets;
    vector<vector<double>> points;
    double bias = 0.0;
    // Hàm kernel
    double kernel(vector<double> &x1, vector<double> &x2)
    {
        double sum = 0.0;
        for (int i = 0; i < x1.size(); ++i)
            sum += x1[i] * x2[i];
        return sum;
    }
    // Hàm SVM output
    double svmOutput(vector<double> x)
    {
        double result = 0.0;
        for (int i = 0; i < alpha.size(); ++i)
        {
            result += alpha[i] * tickets[i] * kernel(points[i], x);
        }
        return result + bias;
    }
    // Hàm kiểm tra và thay đổi
    int takeStep(int i1, int i2)
    {
        if (i1 == i2)
            return 0;
        double alph1 = alpha[i1];
        double alph2 = alpha[i2];
        int y1 = tickets[i1];
        int y2 = tickets[i2];
        double s = y1 * y2;
        double E1 = svmOutput(points[i1]) - y1;
        double E2 = svmOutput(points[i2]) - y2;
        double L, H;
        if (y1 != y2)
        {
            L = max(0.0, alph2 - alph1);
            H = min(C, C + alph2 - alph1);
        }
        else
        {
            L = max(0.0, alph2 + alph1 - C);
            H = min(C, alph2 + alph1);
        }
        if (L == H)
        {
            return 0;
        }
        double k11 = kernel(points[i1], points[i1]);
        double k12 = kernel(points[i1], points[i2]);
        double k22 = kernel(points[i2], points[i2]);
        double eta = k11 + k22 - 2 * k12;
        double a2;
        if (eta > 0)
        {
            a2 = alph2 + y2 * (E1 - E2) / eta;
            if (a2 < L)
                a2 = L;
            else if (a2 > H)
                a2 = H;
        }
        else
        {
            // Tính toán giá trị khách quan tại L và H
            double Lobj = svmOutput(points[i2]) * y2 * L;
            double Hobj = svmOutput(points[i2]) * y2 * H;
            if (Lobj < Hobj - eps)
                a2 = L;
            else if (Lobj > Hobj + eps)
                a2 = H;
            else
                a2 = alph2;
        }
        if (fabs(a2 - alph2) < eps * (a2 + alph2 + eps))
            return 0;
        double a1 = alph1 + s * (alph2 - a2);
        alpha[i1] = a1;
        alpha[i2] = a2;
        // Cập nhật bias (nếu cần)
        double b1 = -E1 - y1 * k11 * (a1 - alph1) - y2 * k12 * (a2 - alph2) + bias;
        double b2 = -E2 - y1 * k12 * (a1 - alph1) - y2 * k22 * (a2 - alph2) + bias;
        if (0 < a1 && a1 < C)
            bias = b1;
        else if (0 < a2 && a2 < C)
            bias = b2;
        else
            bias = (b1 + b2) / 2;

        return 1;
    }
    // Hàm kiểm tra một ví dụ
    int examineExample(int i2)
    {
        int y2 = tickets[i2];
        double alph2 = alpha[i2];
        double E2 = svmOutput(points[i2]) - y2;
        double r2 = E2 * y2;
        if ((r2 < -tol && alph2 < C) || (r2 > tol && alph2 > 0))
        {
            int numAlpha = alpha.size();
            // Lựa chọn i1 bằng heuristic (tùy chỉnh nếu cần)
            int i1 = -1;
            if (numAlpha > 1)
            {
                random_device rd;
                mt19937 gen(rd());
                uniform_int_distribution<> dis(0, numAlpha - 1);
                i1 = dis(gen); // Chọn một số ngẫu nhiên
                if (takeStep(i1, i2) == 1)
                {
                    return 1;
                }
            }
            // Lặp qua tất cả các alpha non-zero và non-C
            for (int i = 0; i < numAlpha; ++i)
            {
                if (alpha[i] != 0 && alpha[i] != C)
                {
                    if (takeStep(i, i2) == 1)
                    {
                        return 1;
                    }
                }
            }
            // Lặp qua tất cả các ví dụ
            for (int i = 0; i < numAlpha; ++i)
            {
                if (takeStep(i, i2) == 1)
                {
                    return 1;
                }
            }
        }
        return 0;
    }
    void smoAlgorithm()
    {
        int numChanged = 0;
        bool examineAll = true;
        int numPoints = points.size();
        while (numChanged > 0 || examineAll)
        {
            numChanged = 0;
            if (examineAll)
            {
                for (int i = 0; i < numPoints; ++i)
                    numChanged += examineExample(i);

                examineAll = false;
            }
            else
            {
                for (int i = 0; i < numPoints; ++i)
                    if (alpha[i] > 0 && alpha[i] < C)
                        numChanged += examineExample(i);

                if (numChanged == 0)
                    examineAll = true;
            }
        }
    }
};
void introduction()
{
    cout << GREEN_TEXT << string(19, ' ') << "|" << string(22, ' ') << WHITE_TEXT << "DE TAI: " << BLUE_TEXT
         << "SUPPORT VECTOR MACHINE" << string(22, ' ') << GREEN_TEXT << "|" << endl;
    cout << string(19, ' ') << GREEN_TEXT << "|" << string(18, ' ') << BLUE_TEXT
         << "Sequential Minimal Optimization Algorithm" << string(15, ' ') << GREEN_TEXT << "|" << endl;
    cout << string(19, ' ') << "|" << string(32, ' ') << CYAN_TEXT
         << "MACHINE
            LEARNING " << string(25, ' ') << GREEN_TEXT << " |
        " << endl;
            cout
            << string(19, ' ') << "|" << string(74, ' ') << "|" << endl;
    cout << string(19, ' ') << "|" << string(24, ' ') << WHITE_TEXT
         << "GIAO
        VIEN HUONG DAN : " << YELLOW_TEXT << " NGUYEN TAN KHOI "
         << string(14, ' ')
         << GREEN_TEXT
         << "|"
         << endl;
    cout << string(19, ' ') << "|" << string(10, ' ') << WHITE_TEXT
         << "SINH
        VIEN THUC HIEN : "
         << string(43, ' ')
         << GREEN_TEXT
         << "|"
         << endl;
    cout << string(19, ' ') << "|" << string(10, ' ') << YELLOW_TEXT << "NGUYEN DOAN THANH HOANG" << string(5, ' ')
         << "LOP 23T_KHDL2" << string(5, ' ') << "NHOM 1" << string(12, ' ') << GREEN_TEXT << "|" << endl;
    cout << string(19, ' ') << "|" << string(10, ' ') << YELLOW_TEXT
         << "PHAN
        VAN HIEU "
         << string(15, ' ') << "LOP 23T_KHDL1" << string(5, ' ') << "NHOM 1" << string(12, ' ') << GREEN_TEXT << "|"
         << endl;
    cout << string(19, ' ') << "|" << string(74, ' ') << "|" << endl;
    cout << string(19, ' ') << "|" << string(32, ' ') << YELLOW_TEXT
         << "K23
            DUT " << string(35, ' ') << GREEN_TEXT << " |\n ";
                                                             cout
                                                             << string(19, ' ') << "|" << string(74, ' ') << "|"
                                                             << endl;
    cout << string(19, ' ') << "|";
    for (int i = 0; i < 74; i++)
        cout << "_";
    cout << "|" << endl;
    cout << CLEAR_TEXT << endl;
}
void display(int Data, SVM svm, vector<double> w)
{
    cout << WHITE_TEXT << string(4, ' ');
    for (int i = 0; i <= 55; i++)
        cout << "_";
    cout << endl;
    cout << string(4, ' ') << "|" << string(10, ' ') << RED_TEXT
         << "KET QUA
            SAU KHI CHAY THUAT TOAN SMO
            " << string(10, ' ') << CLEAR_TEXT << " |\n ";
                                                         // cout << string(4, ' ') << "|" << string(55, ' ') << "|";
                                                         cout
                                                         << string(4, ' ') << "|";
    for (int i = 0; i < 55; i++)
        cout << "_";
    cout << "|\n";
    cout << string(4, ' ') << "|" << string(15, ' ')
         << RED_TEXT "THONG TIN VE
            SUPPORT VECTOR " << CLEAR_TEXT << string(12, ' ') << " |\n ";
                                                                        cout
                                                                        << string(4, ' ') << "|";
    for (int i = 0; i < 55; i++)
        cout << "_";
    cout << "|\n";
    cout << string(4, ' ') << "|" << RED_TEXT << "STT" << CLEAR_TEXT << "|" << string(5, ' ') << CYAN_TEXT
         << "dual coefficient" << CLEAR_TEXT << string(5, ' ') << "|" << string(5, ' ') << CYAN_TEXT << "support vector"
         << CLEAR_TEXT << string(5, ' ') << "|\n";
    cout << string(4, ' ') << "|___|__________________________" << "|" << "________________________|\n";
    for (int i = 0; i < Data; i++)
    {
        if (svm.alpha[i] > eps)
        {
            cout << string(4, ' ') << "|" << CYAN_TEXT << i + 1 << CLEAR_TEXT << " |" << string(9, ' ') << fixed
                 << setprecision(3) << svm.alpha[i] << string(12, ' ') << "|";
            cout << fixed << setprecision(2) << string(3, ' ') << svm.points[i][0] << " " << svm.points[i][0] << " "
                 << svm.points[i][0] << " " << svm.points[i][0] << string(2, ' ') << "|\n";
        }
    }
    cout << string(4, ' ') << "|___|";
    cout << "__________________________|" << "________________________|\n";
    cout << string(4, ' ') << "|" << string(18, ' ') << RED_TEXT
         << "THONG TIN
            VE HYPERPLANE " << CLEAR_TEXT << string(14, ' ') << " |\n ";
                                                                       cout
                                                                       << string(4, ' ')
                                                                       << "|___________________________________________"
                                                                          "____________|\n";
    cout << string(4, ' ') << "|" << string(5, ' ') << WHITE_TEXT << CYAN_TEXT << " vector trong so w " << CLEAR_TEXT
         << string(5, ' ') << "|" << string(5, ' ') << CYAN_TEXT << " HE SO BIAS " << CLEAR_TEXT << string(5, ' ')
         << "|\n";
    cout << string(4, ' ') << "|______________________________|________________________|\n";
    cout << string(4, ' ') << "|" << string(4, ' ');
    for (auto x : w)
    {
        cout << x << " ";
    }
    cout << " " << string(2, ' ') << "|" << string(10, ' ') << svm.bias << string(10, ' ') << "|\n";
    cout << string(4, ' ') << "|______________________________|________________________|\n";
    cout << endl;
}
int main()
{
    introduction();
    vector<Record> records;
    cout << YELLOW_TEXT
         << "a. TIEN HANH DOC FILE TU TEP DU LIEU
\n ";
        createRecordVector(records, "iris.tab");
    // int Data; cin >> Data;
    vector<vector<double>> trainingPoints(records.size(), vector<double>(4)); //
    Tạo 2D vector vector<int> tickets(records.size());
    for (int i = 0; i < records.size(); ++i)
    {
        trainingPoints[i][0] = records[i].sepal_width;
        trainingPoints[i][1] = records[i].sepal_length;
        trainingPoints[i][2] = records[i].petal_width;
        trainingPoints[i][3] = records[i].petal_length;
    }
    string check = records[0].classname;
    for (int i = 0; i < records.size(); i++)
    {
        if (records[i].classname == check)
            tickets[i] = 1;
        else
            tickets[i] = -1;
    }
    SVM svm;
    svm.points = trainingPoints; // Khởi tạo `points`
    svm.tickets = tickets;
    svm.alpha = vector<double>(trainingPoints.size(), 0.0);
    // Chạy thuật toán SMO
    cout << "\nb. TIEN HANH CHAY THUAT TOAN SMO\n";
    svm.smoAlgorithm();
    cout << " DA CHAY XONG THUAT TOAN SMO!!!\n" << endl;
    // Tính toán và hiển thị vector trọng số w
    vector<double> w(svm.points[0].size(), 0.0);
    for (int i = 0; i < svm.alpha.size(); ++i)
    {
        for (int j = 0; j < w.size(); ++j)
            w[j] += svm.alpha[i] * svm.tickets[i] * svm.points[i][j];
    }
    cout << "c. TIEN HANH XUAT KET QUA CUA THUAT TOAN SMO:\n" << endl;
    display(records.size(), svm, w);
    cout << YELLOW_TEXT << "d. KET THUC CHUONG TRINH!!!\n\n";
    cout << string(15, ' ') << CYAN_TEXT
         << "================================== THANK
            YOU !!!=
        = == == == == == == == == == == == == == == == ==\n ";
                                                             cout
                                                             << CLEAR_TEXT;
}
