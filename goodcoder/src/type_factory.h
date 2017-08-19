/// @file type.h
/// @brief 
/// @author sunminqi@baidu.com
/// @version 1.0
/// @date 2017-07-31
#ifndef GOODCODER_SRC_TYPE_H
#define GOODCODER_SRC_TYPE_H

#include"parse_struct.h"
#include"type_spec.h"
class TypeFactory{
public:
    Type* createType(Form str){
        switch(str){
            case INT:
                return new TypeInt(str);
                break;
        }
    }
private:
};
#endif