/// @file type.h
/// @brief 
/// @author sunminqi@baidu.com
/// @version 1.0
/// @date 2017-07-31
#ifndef GOODCODER_SRC_TYPE_H
#define GOODCODER_SRC_TYPE_H

#include <string>
#include "parse_struct.h"
namespace goodcoder{
class Type{
public:
    Type(){
        _str_type = Form::NONE;
        _len = 0;
    }
    virtual ~Type(){};

    Type(const Type&) = delete;
    Type& operator=(const Type&) = delete;
    virtual void print() = 0;
    virtual int set_val(const std::string&) = 0;
protected:
    Form _str_type;
    int _len;
};
}
#endif
