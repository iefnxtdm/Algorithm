/// @file type.h
/// @brief 
/// @author sunminqi@baidu.com
/// @version 1.0
/// @date 2017-07-31
#ifndef GOODCODER_SRC_TYPE_H
#define GOODCODER_SRC_TYPE_H

#include <string>
#include "parse_struct.h"
class Type{
public:
    Type(const std::string& str);
    virtual ~Type(){};

    Type(cont Type&) = delete;
    Type& operator=(const Type&) = delete;
    virtual void print() = override;
private:
    Form _str_type;
    int _len;
};

#endif
