/// @file type_int.h
/// @brief 
/// @author sunminqi@baidu.com
/// @version 1.0
/// @date 2017-07-31
#ifndef GOODCODER_SRC_TYPE_INT_H
#define GOODCODER_SRC_TYPE_INT_H
#include <string>
#include "type.h"
class TypeInt : public type{
public:
    TypeInt(const std::string& str, bool is_arr){
        str_type =  "int";
        _is_array = is_arr;
        
    }
    void print();
private:
    int* var;
}

#endif
