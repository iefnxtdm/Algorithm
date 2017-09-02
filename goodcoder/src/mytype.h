/// @file mytype.h
/// @brief 
/// @author sunminqi@baidu.com
/// @version 1.0
/// @date 2017-07-31
#ifndef GOODCODER_SRC_MYTYPE_H
#define GOODCODER_SRC_MYTYPE_H

#include <string>
#include "parse_struct.h"
namespace goodcoder{
class MyType{
public:
    MyType(){}
    virtual ~MyType(){};

    MyType(const MyType&) = delete;
    MyType& operator=(const MyType&) = delete;
    virtual void print() = 0;
    virtual int set_val(const std::string&) = 0;
    virtual int get_val(void*& val, int& len, Form& form) = 0;
};
}
#endif
