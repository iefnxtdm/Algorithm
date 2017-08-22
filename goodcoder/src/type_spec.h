/// @file type_int.h
/// @brief 
/// @author sunminqi@baidu.com
/// @version 1.0
/// @date 2017-07-31
#ifndef GOODCODER_SRC_TYPE_SPEC_H
#define GOODCODER_SRC_TYPE_SPEC_H

#include<iostream>
#include <string>
#include <vector>
#include <sstream>
#include "type.h"
#include "string_util.h"

namespace goodcoder{
class TypeInt : protected Type{
public:
    TypeInt(const std::string& str, int len){
        _str_type = Form::INT;
        _len = len;
        _var = new int[len];
    }
    ~TypeInt(){
        delete _var;
        _var = nullptr;
    }
    int set_val(const std::string& str){
        std::vector<std::string> tmp;
        int res = 0;
        StringUtil::splits_tring(str, ",", &tmp, 100);
        for(int i = 0; i < tmp.size(); ++i){
            StringUtil::str_to_int(str, res);
            _var[i] = res;
        }
        return 0;
    }
    void print(){
        std::stringstream ss;
        ss << "var=int;len=" << _len << "| ";
        for (int i = 0; i < _len; ++i){
            ss << _var[i] << "\t";
        }
        std::cout << ss.str() << std::endl;
    }
private:
    int* _var;
};

class TypeFloat : protected Type{
public:
    TypeFloat(const std::string& str, int len){
        _str_type = Form::FLOAT;
        _len = len;
        _var = new float[_len];
    }
    ~TypeFloat(){
        delete _var;
        _var = nullptr;
    }
    int set_val(const std::string& str){
        std::vector<std::string> tmp;
        float res = 0;
        StringUtil::splits_tring(str, ",", &tmp, 100);
        for (int i = 0; i < tmp.size(); ++i){
            StringUtil::str_to_float(str, res);
            _var[i] = res;
        }
        return 0;
    }
    void print(){
        std::stringstream ss;
        ss << "var=float;len=" << _len << "| "; 
        for (int i = 0; i < _len; ++i){
            ss << _var[i] << "\t";
        }
        std::cout << ss.str() << std::endl;
    }
private:
    float* _var;
};

class TypeStr : protected Type{
public:
    TypeStr(const std::string& str, int len){
        _str_type = Form::STRING;
        _len = len;
        _var = str;
    }
    void print(){
        std::cout << "var=str | " << _var << std::endl;
    }
private:
    std::string _var;
};
}
#endif

