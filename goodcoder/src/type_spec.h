/// @file type_int.h
/// @brief 
/// @author sunminqi@baidu.com
/// @version 1.0
/// @date 2017-07-31
#ifndef GOODCODER_SRC_TYPE_SPEC_H
#define GOODCODER_SRC_TYPE_SPEC_H

#include <string>
#include "type.h"
#include 
class TypeInt : public Type{
public:
    TypeInt(const std::string& str, int len){
        _str_type = INT;
        _len = len;
        
    }
    void print() ;
private:
    int* var;
};
class TypeFloat : public type{
public:
    TypeFloat(const std::string& str, int len){
        _str_type = Float;
        _len = len;
        
    }
    void print() ;
private:
    float* var;
};
class TypeStr : public type{
public:
    TypeInt(const std::string& str, int len){
        _str_type = STRING;
        _len = len;
        
    }
    void print();
private:
    char* var;
};
#endif
