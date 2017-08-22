/// @file main.cpp
/// @brief
/// @author sunminqi@baidu.com
/// @version 1.0
/// @date 2017-07-31
#include<vector>
#include<fstream>
#include"string_util.h"
#include"parser.h"
using std::fstream;
using std::string;
int main(){
    std::string path = "~/goodcoder/data/data.txt";
    std::ifstream in(path.c_str());
    string line;
    goodcoder::Parser parser;
    while(getline(in, line)){
        if (!parser.parse(line)){
            //TODO warning
        }
    }
    parser.print();
    return 0;
}
