/// @file main.cpp
/// @brief
/// @author sunminqi@baidu.com
/// @version 1.0
/// @date 2017-07-31
#include<vector>
#include<fstream>
#include<iostream>
#include "com_log.h"
#include "Configure.h"
#include"string_util.h"
#include"parser.h"
#include "format_parser.h"
using std::fstream;
using std::string;
using std::cout;
int main(){
    comcfg::Configure conf;
    int ret = 0;
    ret = conf.load("./conf", "parser.conf");
    // load conf
    if (ret != 0){
        cout << "read conf fail" << std::endl;
        return -1;
    }
    // init log
    ret = com_loadlog("./conf", "comlog.conf");
    if (ret != 0){
        cout << "load comlog fail" << std::endl;
        return -1;
    }
    goodcoder::Parser parser;
    std::string path1(conf["data"]["data1"].to_cstr());
    std::string path2(conf["data"]["data2"].to_cstr());
    parser.init(path1);
    goodcoder::FormatParser fp;
    fp.init(path2);
    fp.parse(path2);
    com_closelog(500);
    return 0;
}
